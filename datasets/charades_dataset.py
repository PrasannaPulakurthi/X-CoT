import os
import cv2
import torch
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture
import torchvision


class CHARADESDataset(Dataset):
    """
        videos_dir: directory where all videos are stored
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """

    def __init__(self, config: Config, split_type='train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type

        if config.platform == 'local':
            pth = 'data/Charades/'
        else:
            raise NotImplementedError
        
        if config.test_mode =='debug':
            train_file = pth + 'Charades_v1_train_debug.csv'
            test_file = pth + 'Charades_v1_test_debug.csv'
        elif config.test_mode == 'benchmark':
            train_file = pth + 'Charades_v1_train.csv'
            test_file = pth + 'Charades_v1_test.csv'
        else:
            raise NotImplementedError

        self.vid2cap = {}
        self.all_train_pairs = []
        self.all_test_pairs = []
        if split_type == 'train':
            self.train_vids = []
            self.df_train = pd.read_csv(train_file)
            for idx in range(len(self.df_train)):
                name = self.df_train['id'][idx]
                if name not in self.vid2cap:
                    self.vid2cap[name] = self.df_train['script'][idx]
                    self.all_train_pairs.append([name, self.vid2cap[name]])
            self.idx2vid_name = {idx: name for idx, name in enumerate(self.vid2cap)}

            # self._compute_vid2caption()
            # self._construct_all_train_pairs()

        elif split_type == 'test':
            self.test_vids = []
            self.df_test = pd.read_csv(test_file)
            # print('len(self.df_test))', len(self.df_test))
            # print('set len(self.df_test))', len(set(self.df_test['id'])))
            for idx in range(len(self.df_test)):
                # print('idx', idx)
                # print('self.df_test[id][idx]', self.df_test['id'][idx])
                name = self.df_test['id'][idx]
                if name not in self.vid2cap:
                    self.vid2cap[name] = self.df_test['script'][idx]
                self.all_test_pairs.append([name, self.vid2cap[name]])
            self.idx2vid_name = {idx: name for idx, name in enumerate(self.vid2cap)}

            # self._compute_vid2caption()
            # self._construct_all_test_pairs()

    def load_frames_from_folder(self, path_name):
        video = torch.zeros((0, self.config.num_frames, 1, 3, 224, 224), dtype=torch.double)
        video_folder = path_name.replace('all', 'frames')[:-4]  # + '/'
        # print('video_folder', video_folder)
        total_num_frames = len(glob.glob(video_folder + '/*.jpg'))

        tmp_img_all = []

        if self.config.num_frames < (total_num_frames / 3) or True:
            sample_indx = np.linspace(0 * 3 + 1, total_num_frames, num=self.config.num_frames + 1, dtype=int)
            ranges = []
            for idx, interv in enumerate(sample_indx[:-1]):
                ranges.append((interv, sample_indx[idx + 1] - 1))
            frames_indx = [(x[0] + x[1]) // 2 for x in ranges]

            for tmp_idx in frames_indx:
                check_path = video_folder + '/' + str("{:04d}".format(tmp_idx)) + '.jpg'
                if os.path.exists(check_path):
                    frame = cv2.imread(video_folder + '/' + str("{:04d}".format(tmp_idx)) + '.jpg')
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = torch.from_numpy(frame)
                    # (H x W x C) to (C x H x W)
                    frame = frame.permute(2, 0, 1)
                    tmp_img_all.append(frame)

            if len(tmp_img_all) == self.config.num_frames:
                video = torch.stack(tmp_img_all, dim=0).float() / 255  # .unsqueeze(0)
            else:
                video = torch.stack(tmp_img_all, dim=0).float() / 255  # .unsqueeze(0)
                add_zeros = torch.zeros(self.config.num_frames - len(tmp_img_all), 3, video.shape[2], video.shape[3])
                video = torch.cat((video, add_zeros), 0)

        else:
            print('NOT ENOUGH FRAMES, CHECK WHAT HAPPENS')
            for tmp_idx in range(0, int(total_num_frames / 3)):
                tmp_img = torchvision.io.read_image(
                    video_folder + '/' + str("{:04d}".format(tmp_idx * 3 + 1)) + '.jpg') / 255
                tmp_img_all.append(tmp_img.unsqueeze(0))
            vid_length = len(tmp_img_all)

            if len(tmp_img_all) > self.config.num_frames:
                add_zeros = torch.zeros(self.config.num_frames - len(tmp_img_all), 3, 224, 224)

            video[:, :vid_length] = torch.stack(tmp_img_all, dim=0)

        return video

    def __getitem__(self, index):
        video_path, caption, video_id = self._get_vidpath_and_caption_by_index(index)
        if os.path.isfile(video_path):
            imgs, idxs = VideoCapture.load_frames_from_video(video_path,
                                                             self.config.num_frames,
                                                             self.config.video_sample_type,)
                                                             # self.config.cut_video)
        else:
            imgs = self.load_frames_from_folder(video_path)

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
        return len(self.all_test_pairs)

    def _get_vidpath_and_caption_by_index(self, index):
        # returns video path and caption as string
        if self.split_type == 'train':
            vid = self.idx2vid_name[index]
            caption = self.vid2cap[vid]
            # vid, caption = self.all_train_pairs[index]
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
        else:
            vid = self.idx2vid_name[index]
            caption = self.vid2cap[vid]
            # vid, caption = self.all_test_pairs[index]
            video_path = os.path.join(self.videos_dir, vid + '.mp4')

        return video_path, caption, vid

    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        for vid in self.train_vids:
            for caption in self.vid2caption[vid]:
                self.all_train_pairs.append([vid, caption])

    def _construct_all_test_pairs(self):
        self.all_test_pairs = []
        for vid in self.test_vids:
            for caption in self.vid2caption[vid]:
                self.all_test_pairs.append([vid, caption])

    def _compute_vid2caption(self):
        self.vid2caption = defaultdict(list)
        if self.split_type == 'train':
            videoids = self.train_vids
            for idx, video_id in enumerate(videoids):
                caption = self.df_train[video_id]['script']
                self.vid2caption[video_id].append(caption)
        elif self.split_type == 'test':
            videoids = self.test_vids
            for video_id in videoids:
                caption = self.df_test[video_id]['script']
                self.vid2caption[video_id].append(caption)
