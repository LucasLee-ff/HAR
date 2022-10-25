from torch.utils.data import Dataset
import torch
import numpy as np
import os
from dataset_utils import load_video, downsample, random_sample, center_sample, normalize


class DarkVid(Dataset):
    def __init__(self, root, mode='train', clip_len=16, transform=None, multi_scale=None):
        self.root = root
        self.mode = mode
        self.clip_len = clip_len
        self.transform = transform
        self.multi_scale = multi_scale
        self.vid_list = []
        self.label_list = []
        with open(os.path.join(root, mode + '.txt')) as f:
            lines = f.readlines()
            for line in lines:
                n, label, vid_name = line.split()
                self.vid_list.append(vid_name)
                self.label_list.append(int(label))

    def __getitem__(self, idx):
        vid_path = os.path.join(self.root, self.mode, self.vid_list[idx])
        buffer = load_video(vid_path)
        # if the total number of frames in the video is larger than 64, downsample along the temporal axis by 2 first.
        if len(buffer) >= 2 * self.clip_len:
            if self.mode == 'train':
                if np.random.random() < 0.5:
                    buffer = downsample(buffer, start=0)
                else:
                    buffer = downsample(buffer, start=1)
            else:
                buffer = downsample(buffer, start=0)

        if self.mode == 'train':
            buffer = random_sample(buffer, self.clip_len)
        else:
            buffer = center_sample(buffer, self.clip_len)
        buffer = buffer / 255.0
        buffer = normalize(buffer)
        buffer = buffer.transpose((3, 0, 1, 2))  # (T, H, W, C) -> (C, T, H, W)
        buffer = torch.from_numpy(buffer)
        if self.transform:
            buffer = self.transform(buffer)
        if self.multi_scale:
            buffer2 = self.multi_scale(buffer)
            return buffer, buffer2, torch.tensor(self.label_list[idx])
        else:
            return buffer, torch.tensor(self.label_list[idx])

    def __len__(self):
        return len(self.label_list)
