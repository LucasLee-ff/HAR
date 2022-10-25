import cv2
import numpy as np


def load_video(path):
    # Load all frames of the video
    vid = cv2.VideoCapture(path)
    if not vid.isOpened():
        raise Exception('no video in {}'.format(path))
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


def normalize(buffer):
    # Normalize every frame using "mean" and "standard deviation from ARID dataset"
    for i, frame in enumerate(buffer):
        frame = (frame - np.array([[[0.079612, 0.073888, 0.072454]]])) / np.array([[[0.100459, 0.0970497, 0.089911]]])
        buffer[i] = frame
    return buffer


def downsample(buffer, start=0, interval=2):
    # downsample the video along the temporal axis with step=interval
    return buffer[start::interval, :, :, :]


def random_sample(buffer, clip_len):
    # get clip_len consecutive frames start from a randomly selected index
    length = len(buffer)
    if clip_len > length:
        raise Exception("clip_len too large!")
    start_idx = np.random.randint(0, length - clip_len + 1)
    return buffer[start_idx:start_idx + clip_len, :, :, :]


def center_sample(buffer, clip_len):
    length = len(buffer)
    if clip_len > length:
        raise Exception("clip_len too large!")
    center_idx = length // 2
    offset = clip_len // 2
    return buffer[center_idx - offset:center_idx + offset, :, :, :]

