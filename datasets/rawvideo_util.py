import torch as th
import numpy as np
from PIL import Image
# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# pip install opencv-python
import cv2
import torch
import random
import time
import os

def convert_to_rgb(image):
    return image.convert("RGB")

class RawVideoExtractorCV2():
    def __init__(self, centercrop=False, size=224, framerate=-1, ):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.transform = self._transform(self.size)

    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            convert_to_rgb,
            ToTensor(),
            # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def video_to_tensor_xpool(self, video_path, num_frames, sample, start_time, end_time, cut_video=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            time.sleep(1)
            cap = cv2.VideoCapture(video_path)
            print('cannot open', video_path)

        assert cap.isOpened()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_duration = frame_count / fps
        # print('frame_count', frame_count)
        # print('fps', fps)
        # print('total_duration', total_duration)

        end_time = min(end_time, total_duration)
        vlen = (end_time - start_time) * fps
        # print('start end', start_time, end_time)
        # print('vlen', vlen)

        # get indexes of sampled frames
        acc_samples = min(num_frames, vlen)

        # # cut video after 10 sec
        # stop_value = 10 * fps if cut_video and 10 * fps < vlen else vlen

        intervals = np.linspace(start=start_time*fps, stop=end_time*fps, num=acc_samples + 1).astype(int)
        ranges = []

        # ranges constructs equal spaced intervals (start, end)
        # we can either choose a random image in the interval with 'rand'
        # or choose the middle frame with 'uniform'
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))

        if sample == 'rand':
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        else:  # sample == 'uniform':
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

        frames = []
        filename = os.path.splitext(os.path.basename(video_path))[0]
        # print('frame_idxs', frame_idxs)
        for index in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            # print(ret, index)
            if not ret:
                n_tries = 5
                for _ in range(n_tries):
                    ret, frame = cap.read()
                    if ret:
                        break
            if ret:
                cv2.imwrite(f'images/{filename}_{index}.jpg', frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                # (H x W x C) to (C x H x W)
                frame = frame.permute(2, 0, 1)
                frames.append(frame)
            else:
                if len(frames) > 0:
                    frames.append(frames[-1].clone())       # Different from previous version, just append the last frame. One example is
                else:
                    frames.append(torch.zeros(3, 224, 224))
                # /mnt/search_efs/public_datasets/didemo/all_videos/25566432@N00_2406759068_87a681e6db.mov, meta-data says it has 82s, but
                # it actually only has 23s.
                # raise ValueError

        while len(frames) < num_frames:
            frames.append(frames[-1].clone())

        frames = torch.stack(frames).float() / 255
        cap.release()
        return frames, frame_idxs

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) and -1 < start_time < end_time
        assert sample_fp > -1

        # Samples a frame sample_fp X frames.
        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration

        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0:
            interval = 1

        inds = [ind for ind in np.arange(0, fps, interval)]
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]

        ret = True
        images, included = [], []

        for sec in np.arange(start_sec, end_sec + 1):
            if not ret:
                break
            sec_base = int(sec * fps)
            for ind in inds:
                cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

        cap.release()

        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)
        return {'video': video_data}

    def get_video_data(self, video_path, start_time=None, end_time=None, sample_type=None, num_frames=12):
        if sample_type is not None:
            frames, frame_idxs = self.video_to_tensor_xpool(video_path, num_frames, sample=sample_type,
                                                     start_time=start_time, end_time=end_time)
            return frames
        else:
            image_input = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate,
                                               start_time=start_time, end_time=end_time)
            return image_input

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data


# An ordinary video frame extractor based CV2
RawVideoExtractor = RawVideoExtractorCV2
