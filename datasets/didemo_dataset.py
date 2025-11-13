import torch
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.rawvideo_util import RawVideoExtractor
import csv

class DiDeMoDataset(Dataset):
    """
        videos_dir: directory where all videos are stored
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """

    def __init__(self, config: Config, split_type='train', img_transforms=None):
        self.config = config
        self.img_transforms = img_transforms
        self.split_type = split_type
        self.video_dir = config.videos_dir

        
        if self.config.platform == 'local':
            pth = 'data/Didemo/'
        else:
            raise NotImplementedError
        
        if split_type == 'train':
            if self.config.train_mode == 'debug':
                self.label_csv = pth + 'didemo_train_label_debug.csv'
            elif self.config.train_mode == 'benchmark':
                self.label_csv = pth + 'didemo_train_label.csv'
            else:
                raise NotImplementedError

        elif split_type == 'test':
            if self.config.test_mode =='debug':
                self.label_csv = pth + 'didemo_test_label_debug.csv'
            elif self.config.test_mode =='benchmark':
                self.label_csv = pth + 'didemo_test_label.csv'
            else:
                raise NotImplementedError

        else:
            print('unseen data split type!')
            raise NotImplementedError

        # self.label_csv = pth + 'didemo_{}_label.csv'.format(split_type)
        self.load_frames_from_preprocess_pth = False
        self.num_frames = config.num_frames
        self.rawVideoExtractor = RawVideoExtractor(framerate=1, size=224)       # frame rate and size are both dummy
        
        if split_type == 'train':
            self._construct_all_train_pairs()
        else:
            self._construct_all_test_pairs()

    def _get_rawvideo(self, video_path, s, e):
        # imgs_from_video = []
        # for i in range(len(s)):
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0. else 0.
        end_time = end_time if end_time >= 0. else 0.
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = end_time + 1

        cache_id = "{}_{}_{}".format(video_path, start_time, end_time)
        # Should be optimized by gathering all asking of this video
        # print('video_path', video_path, 'start_time', start_time, 'end_time', end_time)
        imgs = self.rawVideoExtractor.get_video_data(video_path, start_time, end_time,
                                                     sample_type='uniform', num_frames=self.num_frames)
        # print(imgs.shape)
            # imgs_from_video.append(imgs)
        return imgs

    def __getitem__(self, index):
        video_id, caption, formatted_data_path, starts, ends, video_path = self._get_vidpath_and_caption_by_index(index)
        starts = int(starts.replace('tensor', '').replace('[', '').replace(']', '').replace(')', '').replace('(', ''))
        ends = int(ends.replace('tensor', '').replace('[', '').replace(']', '').replace(')', '').replace('(', ''))

        # print(f'video_path={video_path}')
        # video_path=/home/ubuntu/search_efs/public_datasets/didemo/all_videos/32005048@N06_5252757341_6749a9a2bd.mpg
        video_path = self.video_dir + video_path.split('/')[-1]
        # print(f'video_path={video_path}')

        if self.load_frames_from_preprocess_pth:
            formatted_data = torch.load(formatted_data_path)
            imgs = formatted_data['frames'].squeeze()
        else:
            # print(type(starts), starts, ends)
            imgs = self._get_rawvideo(video_path, starts, ends)
        #     print('imgs', imgs.shape)

        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        return {
            'video_id': video_id,
            'video': imgs,
            'text': caption,
        }

    def __len__(self):
        if self.split_type == 'train':
            print(f'>>>length={len(self.all_train_pairs)}')
            return len(self.all_train_pairs)
        print(f'>>>length={len(self.all_test_pairs)}')
        return len(self.all_test_pairs)

    def _get_vidpath_and_caption_by_index(self, index):
        
        if self.split_type == 'train':
            data_id, text, formatted_data_path, starts, ends, video_path = self.all_train_pairs[index]
            data_id = '_'.join(data_id.split('_')[:-1] + ['{:05d}'.format(int(data_id.split('_')[-1]))])
            return data_id, text, formatted_data_path, starts, ends, video_path
        
        else:
            data_id, text, formatted_data_path, starts, ends, video_path = self.all_test_pairs[index]
            data_id = '_'.join(data_id.split('_')[:-1] + ['{:05d}'.format(int(data_id.split('_')[-1]))])
            return data_id, text, formatted_data_path, starts, ends, video_path

    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        with open(self.label_csv, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                if i > 0:
                    success, video_path, starts, ends, text, formatted_data = row
                    data_id = formatted_data.split('/')[-1].split('.')[0]
                    self.all_train_pairs.append([data_id, text, formatted_data, starts, ends, video_path])

    def _construct_all_test_pairs(self):
        self.all_test_pairs = []
        with open(self.label_csv, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                if i > 0:
                    success, video_path, starts, ends, text, formatted_data = row
                    data_id = formatted_data.split('/')[-1].split('.')[0]
                    self.all_test_pairs.append([data_id, text, formatted_data, starts, ends, video_path])



