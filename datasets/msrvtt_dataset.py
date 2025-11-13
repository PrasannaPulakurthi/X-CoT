import os
import cv2
import sys
import torch
import random
import itertools
import numpy as np
import pandas as pd
import ujson as json
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


class MSRVTTDataset(Dataset):
    """
        videos_dir: directory where all videos are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config: Config, split_type = 'train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        self.platform = config.platform
        self.test_mode = config.test_mode

        
        if self.platform == 'local':
            dir = 'data/MSRVTT'
        else:
            raise NotImplementedError
        
        if self.test_mode == 'benchmark':
            test_file_pth = '/MSRVTT_JSFUSION_test.csv'
        elif self.test_mode == 'debug':
            test_file_pth = '/MSRVTT_JSFUSION_test_debug.csv'
        else:
            raise NotImplementedError
        
        test_csv = dir + test_file_pth
        # test_csv = 'data/MSRVTT/MSRVTT_JSFUSION_test.csv'

        db_file = dir + '/MSRVTT_data.json'
        # db_file = 'data/MSRVTT/MSRVTT_data.json'

        if config.msrvtt_train_file == '7k':
            train_csv = dir + '/MSRVTT_train.7k.csv'
        else:
            train_csv = dir + '/MSRVTT_train.9k.csv'

        self.db = load_json(db_file)
        if split_type == 'train':
            train_df = pd.read_csv(train_csv)
            self.train_vids = train_df['video_id'].unique()
            self._compute_vid2caption()
            self._construct_all_train_pairs()
        else:
            self.test_df = pd.read_csv(test_csv)

            
    def __getitem__(self, index):
        video_path, caption, video_id = self._get_vidpath_and_caption_by_index(index)
        imgs, idxs = VideoCapture.load_frames_from_video(video_path, 
                                                         self.config.num_frames, 
                                                         self.config.video_sample_type)

        # process images of video
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        return {
            'video_id': video_id,
            'video': imgs,
            'text': caption,
        }

    
    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.test_df)


    def _get_vidpath_and_caption_by_index(self, index):
        # returns video path and caption as string
        if self.split_type == 'train':
            vid, caption = self.all_train_pairs[index]
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
        else:
            vid = self.test_df.iloc[index].video_id
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
            caption = self.test_df.iloc[index].sentence

        return video_path, caption, vid

    
    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        if self.split_type == 'train':
            for vid in self.train_vids:
                for caption in self.vid2caption[vid]:
                    self.all_train_pairs.append([vid, caption])

            
    def _compute_vid2caption(self):
        self.vid2caption = defaultdict(list)
        for annotation in self.db['sentences']:
            caption = annotation['caption']
            vid = annotation['video_id']
            self.vid2caption[vid].append(caption)
