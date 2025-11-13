from config.base_config import Config
import numpy as np
import torch
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, sim_matrix_inference, generate_embeds_per_video_id, compute_metrics
from tqdm import tqdm
import csv
import json
from modules.reranking_utils import save_ranking_info, load_ranking_info, compute_reranked_recalls

class Trainer(BaseTrainer):
    """
    Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader, 
                 valid_data_loader, tokenizer, lr_scheduler=None, writer=None):

        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer 

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps-1, self.evals_per_epoch+1, dtype=int)[1:]
        
        for batch_idx, data in enumerate(self.train_data_loader):
            # then assume we must tokenize the input, e.g. its a string
            if self.tokenizer is not None:
                data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                              truncation=True)
            if isinstance(data['text'], torch.Tensor):
                data['text'] = data['text'].to(self.device)
            else:
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
            
            data['video'] = data['video'].to(self.device)

            text_embeds, video_embeds_pooled = self.model(data)
            output = sim_matrix_training(text_embeds, video_embeds_pooled, self.pooling_type)
            
            loss = self.loss(output, self.model.clip.logit_scale)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))

            self.global_step += 1
            if self.writer is not None:
                self.writer.add_scalar('train/loss_train', loss.detach().item(), self.global_step)

            total_loss += loss.detach().item()

            if batch_idx % self.log_step == 0:
                print('Train Epoch: {} dl: {}/{} Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    num_steps-1,
                    loss.detach().item()))

            if batch_idx in eval_steps:
                val_res = self._valid_epoch_step(epoch, batch_idx, num_steps-1)
                self.model.train()

                if val_res['R1-window'] > self.best_window:
                    self.best_window = val_res['R1-window']
                    self._save_checkpoint(epoch, save_best=True)

                if val_res['R1'] > self.best:
                    self.best = val_res['R1']

                print(" Current Best Window Average R@1 is {}".format(self.best_window))
                print(" Current Best R@1 is {}\n\n".format(self.best))

        res = {
            'loss_train':  total_loss / num_steps
        }

        return res

    
    def _valid_epoch_step(self, epoch, step, num_steps):
        """
        Validate at a step when training an epoch at a certain step
        :return: A log that contains information about validation
        """
        self.model.eval()
        total_val_loss = 0.0
        text_embed_arr = []
        vid_embed_arr = []
        all_vid_ids = []
        all_texts = []
        
        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):   
                for text in data['text']:
                    all_texts.append(text)

                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(self.device)
                else:
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}

                data['video'] = data['video'].to(self.device)
                
                text_embed, vid_embed, vid_embed_pooled = self.model(data, return_all_frames=True)
                text_embed_arr.append(text_embed.cpu())
                vid_embed_arr.append(vid_embed.cpu())
                sims_batch = sim_matrix_training(text_embed, vid_embed_pooled, self.pooling_type)

                curr_loss = self.loss(sims_batch, self.model.clip.logit_scale)
                total_val_loss += curr_loss.item()

                for v_id in data['video_id']:
                    all_vid_ids.append(v_id)
                
            text_embeds = torch.cat(text_embed_arr)
            vid_embeds = torch.cat(vid_embed_arr)

            # Since we have all pairs, remove duplicate videos when there's multiple captions per video
            vid_embeds_per_video_id = {}
            for idx, v_id in enumerate(all_vid_ids):
                if v_id not in vid_embeds_per_video_id:
                    vid_embeds_per_video_id[v_id] = vid_embeds[idx]
            
            vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])
            
            # Pool frames for inference once we have all texts and videos
            self.model.pool_frames.cpu()
            vid_embeds_pooled = self.model.pool_frames(text_embeds, vid_embeds)
            self.model.pool_frames.cuda()
            # print(text_embeds.size(), vid_embeds.size(), vid_embeds_pooled.size())
            
            '''
            # Pool frames for inference once we have all texts and videos # Move pooling function to GPU before use
            self.model.pool_frames.to(self.device)
            vid_embeds_pooled, _ = self.model.pool_frames([], vid_embeds.to(self.device))
            vid_embeds_pooled = vid_embeds_pooled.cpu()
            '''

            text_embeds_per_video_id, vid_embeds_pooled_per_video_id, cap_per_vid = generate_embeds_per_video_id(text_embeds, 
                    vid_embeds_pooled, all_vid_ids, self.pooling_type)
            # print(text_embeds_per_video_id.size(), vid_embeds_pooled_per_video_id.size())

            sims = sim_matrix_inference(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, self.pooling_type)
            # print(sims.size())

            if self.config.retrieve_topk:
                # Case 1: every video has exactly one caption
                if sims.shape[1] == 1:     
                    flat_sims = sims.squeeze(1)
                
                # Case 2: every video has different number of captions
                else:
                    kept = []
                    for vid_idx, real_len in enumerate(cap_per_vid):
                        kept.append(sims[vid_idx, :real_len, :])       # [C_i, V]
                    flat_sims = torch.cat(kept, dim=0)                 # [num_text, V]

                json_path = f"outputs/{self.config.dataset_name}/{self.config.topk_retrieval_method}_ranking_{self.config.test_mode}.jsonl"
                save_ranking_info(self.config,flat_sims,all_vid_ids,all_texts,json_path)
                results = load_ranking_info(json_path)
                query_ids = [entry["video_index"] for entry in results]
                ranking_indices = [entry["ranking_indices"] for entry in results]
                res = compute_reranked_recalls(query_ids, ranking_indices)

                # Compute window metrics
                for m in res:
                    self.window_metric[m].append(res[m])

                # Compute average of window metrics
                for m in self.window_metric:
                    res[m + "-window"] = np.mean(self.window_metric[m])

                print(f"-------------------------------------\n",
                    f"R@1: {res['R1']} (window: {res['R1-window']})\n", 
                    f"R@5: {res['R5']} (window: {res['R5-window']})\n", 
                    f"R@10: {res['R10']} (window: {res['R10-window']})\n",
                    f"R@20: {res['R20']} (window: {res['R20-window']})\n",
                    f"R@50: {res['R50']} (window: {res['R50-window']})\n",
                    f"R@100: {res['R100']} (window: {res['R100-window']})\n",             
                    f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
                    f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n",
                    "-------------------------------------\n")
                
            total_val_loss = total_val_loss / len(self.valid_data_loader)

            metrics = self.metrics
            res = metrics(sims)
            
            # Compute window metrics
            for m in res:
                self.window_metric[m].append(res[m])

            # Compute average of window metrics
            for m in self.window_metric:
                res[m + "-window"] = np.mean(self.window_metric[m])

            print(f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                  f"R@1: {res['R1']} (window: {res['R1-window']})\n", 
                  f"R@5: {res['R5']} (window: {res['R5-window']})\n", 
                  f"R@10: {res['R10']} (window: {res['R10-window']})\n",
                  f"R@20: {res['R20']} (window: {res['R20-window']})\n",
                  f"R@50: {res['R50']} (window: {res['R50-window']})\n",
                  f"R@100: {res['R100']} (window: {res['R100-window']})\n",             
                  f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
                  f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n",
                  f"Loss: {total_val_loss}")
            
            res['loss_val'] =  total_val_loss

            if self.writer is not None:
                for m in res:
                    self.writer.add_scalar(f'val/{m}', res[m], self.global_step)

            return res