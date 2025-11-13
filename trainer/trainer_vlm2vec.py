from config.base_config import Config
import numpy as np
import torch
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, sim_matrix_inference, generate_embeds_per_video_id, compute_metrics
from tqdm import tqdm
from config.all_config import gen_log
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
        self.config = config
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer 

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0


    def _valid_epoch_step(self, epoch, step, num_steps):
        """
        Validate at a step when training an epoch at a certain step
        :return: A log that contains information about validation
        """
        gen_log(model_path=self.config.model_path, log_name='log_trntst', msg='Start testing')

        self.model.eval()
        text_embed_arr = []
        vid_embed_arr = []
        all_vid_ids = []
        vid_embed_dict = {}
        all_texts = []
        
        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):
                for text in data['text']:
                    all_texts.append(text)
                
                ### Encode Texts
                text_embed = self.model.encode_text(data['text'])  
                text_embed_arr.append(text_embed.cpu())
                
                if self.config.dataset_name == 'MSVD':
                    ### Process Unique Videos Inline
                    for v_id, video in zip(data['video_id'], data['video']):
                        all_vid_ids.append(v_id)  # Keep order for later retrieval
                        if v_id not in vid_embed_dict:  # Process only if video is new
                            vid_embed, _ = self.model.encode_video(video.unsqueeze(0), return_all_frames=True)  # Add batch dim
                            vid_embed_dict[v_id] = vid_embed.squeeze(0).cpu()  # Store unique video embedding
                            
                else:
                    for v_id in data['video_id']:
                        all_vid_ids.append(v_id)

                    vid_embed, _ = self.model.encode_video(data['video'], return_all_frames=True)
                    vid_embed_arr.append(vid_embed.cpu())
                

            ### Reconstruct Video Embeddings for Each Caption
            text_embeds = torch.cat(text_embed_arr)
            
            if self.config.dataset_name == 'MSVD':
                ### Retrieve Video Embeddings in Caption Order
                vid_embeds_list = []
                for v_id in all_vid_ids:
                    if v_id in vid_embed_dict:
                        vid_embeds_list.append(vid_embed_dict[v_id])
                    else:
                        print(f"WARNING: Missing video embedding for {v_id}!")  # Catch missing video errors
                vid_embeds = torch.stack(vid_embeds_list)

            else:
                vid_embeds = torch.cat(vid_embed_arr)

            # Since we have all pairs, remove duplicate videos when there's multiple captions per video
            vid_embeds_per_video_id = {}
            for idx, v_id in enumerate(all_vid_ids):
                if v_id not in vid_embeds_per_video_id:
                    vid_embeds_per_video_id[v_id] = vid_embeds[idx]
            vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])

            # Pool frames for inference once we have all texts and videos # Move pooling function to GPU before use
            self.model.pool_frames.to(self.device)
            vid_embeds_pooled, _ = self.model.pool_frames([], vid_embeds.to(self.device))
            vid_embeds_pooled = vid_embeds_pooled.cpu()

            # print(f'>>>vid_embeds_pooled={vid_embeds_pooled.dtype}') # >>>vid_embeds_pooled=torch.bfloat16

            ### Generate Final Embeddings and Compute Similarity
            text_embeds_per_video_id, vid_embeds_pooled_per_video_id, cap_per_vid = generate_embeds_per_video_id(text_embeds, 
                    vid_embeds_pooled, all_vid_ids, self.pooling_type)
            # print(f'>>>text_embeds_per_video_id={text_embeds_per_video_id.dtype}, vid_embeds_pooled_per_video_id={vid_embeds_pooled_per_video_id.dtype}')
            # >> > text_embeds_per_video_id = torch.float32, vid_embeds_pooled_per_video_id = torch.bfloat16
            
            text_embeds_per_video_id = text_embeds_per_video_id.to(torch.float32)
            vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.to(torch.float32)
            sims = sim_matrix_inference(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, self.pooling_type)
            # total_val_loss = total_val_loss / len(self.valid_data_loader)

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

                json_path = f"outputs/{self.config.dataset_name}/vlm2vec_ranking_{self.config.test_mode}.jsonl"
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
                
            ### Compute Metrics
            metrics = self.metrics
            res = metrics(sims)
            
            ### Compute window metrics
            for m in res:
                self.window_metric[m].append(res[m])

            # Compute average of window metrics
            for m in self.window_metric:
                res[m + "-window"] = np.mean(self.window_metric[m])

            msg = (f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                  f"R@1: {res['R1']} (window: {res['R1-window']})\n", 
                  f"R@5: {res['R5']} (window: {res['R5-window']})\n", 
                  f"R@10: {res['R10']} (window: {res['R10-window']})\n",
                  f"R@20: {res['R20']} (window: {res['R20-window']})\n",
                  f"R@50: {res['R50']} (window: {res['R50-window']})\n",
                  f"R@100: {res['R100']} (window: {res['R100-window']})\n",    
                  f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
                  f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n")
            gen_log(model_path=self.config.model_path, log_name='log_trntst', msg=msg)

            return res
