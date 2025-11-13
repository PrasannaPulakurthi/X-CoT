import os
import argparse
from config.base_config import Config
from modules.basic_utils import mkdirp, deletedir
import datetime
import logging
import numpy as np
import time



def gen_log(model_path, msg, log_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/'+ log_name + '.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(msg)
    logger.removeHandler(fh)
    logger.removeHandler(ch)
    # return logger


class AllConfig(Config):
    def __init__(self):
        super().__init__()

    def time2file_name(self, time):
        year = time[0:4]
        month = time[5:7]
        day = time[8:10]
        hour = time[11:13]
        minute = time[14:16]
        second = time[17:19]
        time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
        return time_filename
    
    def parse_args(self):
        description = 'Text-to-Video Retrieval'
        parser = argparse.ArgumentParser(description=description)
        
        # data parameters
        parser.add_argument('--dataset_name', type=str, default='MSRVTT', help="Dataset name")
        parser.add_argument('--videos_dir', type=str, default='data/MSRVTT/vids', help="Location of videos")
        parser.add_argument('--msrvtt_train_file', type=str, default='9k')
        parser.add_argument('--num_frames', type=int, default=12)
        parser.add_argument('--video_sample_type', default='uniform', help="'rand'/'uniform'")
        parser.add_argument('--input_res', type=int, default=224)

        # experiment parameters
        parser.add_argument('--exp_name', type=str, required=True, help="Name of the current experiment")
        parser.add_argument('--output_dir', type=str, default='./outputs')
        parser.add_argument('--save_every', type=int, default=1, help="Save model every n epochs")
        parser.add_argument('--log_step', type=int, default=10, help="Print training log every n steps")
        parser.add_argument('--evals_per_epoch', type=int, default=10, help="Number of times to evaluate per epoch")
        parser.add_argument('--load_epoch', type=int, help="Epoch to load from exp_name, or -1 to load model_best.pth")
        parser.add_argument('--eval_window_size', type=int, default=5, help="Size of window to average metrics")
        parser.add_argument('--metric', type=str, default='t2v', help="'t2v'/'v2t'")

        # model parameters
        parser.add_argument('--huggingface', action='store_true', default=False)
        parser.add_argument('--arch', type=str, default='clip_transformer')
        parser.add_argument('--clip_arch', type=str, default='ViT-B/32', help="CLIP arch. only when not using huggingface")
        parser.add_argument('--embed_dim', type=int, default=512, help="Dimensionality of the model embedding")

        # training parameters
        parser.add_argument('--loss', type=str, default='clip')
        parser.add_argument('--clip_lr', type=float, default=1e-6, help='Learning rate used for CLIP params')
        parser.add_argument('--noclip_lr', type=float, default=1e-5, help='Learning rate used for new params')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_epochs', type=int, default=5)
        parser.add_argument('--weight_decay', type=float, default=0.2, help='Weight decay')
        parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Warmup proportion for learning rate schedule')

        # frame pooling parameters
        parser.add_argument('--pooling_type', type=str)
        parser.add_argument('--k', type=int, default=10, help='K value for topk pooling')
        parser.add_argument('--attention_temperature', type=float, default=0.01, help='Temperature for softmax (used in attention pooling only)')
        parser.add_argument('--num_mha_heads', type=int, default=1, help='Number of parallel heads in multi-headed attention')
        parser.add_argument('--transformer_dropout', type=float, default=0.3, help='Dropout prob. in the transformer pooling')

        # system parameters
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--seed', type=int, default=24, help='Random seed')
        parser.add_argument('--no_tensorboard', action='store_true', default=False)
        parser.add_argument('--tb_log_dir', type=str, default='logs')

        # added
        parser.add_argument("--datetime",type=str,default=None,help='to be specificed for loading pre-trained checkpoint ')
        parser.add_argument("--platform", type=str, default="local")
        parser.add_argument("--train_mode", type=str, default="benchmark", choices=['debug', 'benchmark'])
        parser.add_argument("--test_mode", type=str, default="benchmark", choices=['debug', 'benchmark'])
        parser.add_argument('--gpu', type=str, default=None, help="gpu id")
        parser.add_argument('--retrieve_topk', action='store_true', default=False)
        parser.add_argument('--topk', type=int, default=20, help='Top K')
        parser.add_argument('--data_no_norm', action='store_true', default=False)
        parser.add_argument('--llm_input_options', type=str, default='image_frames')
        parser.add_argument('--sort_type', type=str, default='sliding_window_parallel')
        parser.add_argument('--passes', type=int, default=10, help='passes')
        parser.add_argument('--topk_retrieval_method', type=str, default='xpool', choices=['xpool', 'vlm2vec', 'clip'])

        args = parser.parse_args()

        if args.datetime is None:
            rand_wait = np.random.randint(low=1, high=20)
            time.sleep(rand_wait)
            date_time = str(datetime.datetime.now())
            date_time = self.time2file_name(date_time)
        else:
            date_time = args.datetime

        args.model_path = os.path.join(args.output_dir, args.exp_name)
        args.tb_log_dir = os.path.join(args.tb_log_dir, args.exp_name, date_time)

        mkdirp(args.model_path)
        mkdirp(args.tb_log_dir)

        return args
