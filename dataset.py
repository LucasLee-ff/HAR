from torch.utils.data import Dataset
import torch
import numpy as np
import os
import cv2


class DarkVid(Dataset):
    def __init__(self, root, mode='train', clip_len=16, transform=None):
        self.root = root
        self.mode = mode
        self.clip_len = clip_len
        self.transform = transform
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
        buffer = self.load_video(vid_path)
        buffer = self.downsample(buffer)
        if self.mode == 'train':
            buffer = self.random_sample(buffer)
        else:
            buffer = self.center_sample(buffer)
        buffer = buffer / 255.0
        buffer = self.normalize(buffer)
        buffer = buffer.transpose((3, 0, 1, 2))  # (T, H, W, C) -> (C, T, H, W)
        buffer = torch.from_numpy(buffer)
        if self.transform:
            buffer = self.transform(buffer)
        return buffer, torch.tensor(self.label_list[idx])

    def __len__(self):
        return len(self.label_list)

    def load_video(self, path):
        # Load all frames of the video
        vid = cv2.VideoCapture(path)
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        buffer = np.empty((total_frames, height, width, 3), np.dtype('float32'))
        count = 0
        while True:
            ret, frame = vid.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buffer[count] = frame
            count += 1
        vid.release()
        return buffer

    def downsample(self, buffer, interval=2):
        # downsample the video along the temporal axis with step=interval
        if self.mode == 'train':
            if np.random.random() < 0.5:
                return buffer[1::interval, :, :, :]
        return buffer[::interval, :, :, :]

    def random_sample(self, buffer):
        # get clip_len=16 consecutive frames start from a randomly selected index
        length = len(buffer)
        start_idx = np.random.randint(0, length - self.clip_len + 1)
        try:
            return buffer[start_idx:start_idx + self.clip_len, :, :, :]
        except:
            raise Exception("clip_len too large!")

    def center_sample(self, buffer):
        length = len(buffer)
        center_idx = length // 2
        offset = self.clip_len // 2
        try:
            return buffer[center_idx - offset:center_idx + offset, :, :, :]
        except:
            raise Exception("clip_len too large!")

    def normalize(self, buffer):
        # Normalize every frame using "mean" and "standard deviation from ARID dataset"
        for i, frame in enumerate(buffer):
            frame = (frame - np.array([[[0.079612, 0.073888, 0.072454]]])) / np.array([[[0.100459, 0.0970497, 0.089911]]])
            buffer[i] = frame
        return buffer
