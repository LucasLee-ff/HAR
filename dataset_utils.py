import cv2
import numpy as np
import math


def load_video(path):
    # Load all frames of the video
    vid = cv2.VideoCapture(path)
    if not vid.isOpened():
        raise Exception('no video in {}'.format(path))
    #total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    #height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames = []
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.max() == 0:
            continue
        frames.append(frame)
    vid.release()
    buffer = np.stack(frames)
    return buffer


def normalize(buffer, mean=[[[0.079612, 0.073888, 0.072454]]], std=[[[0.100459, 0.0970497, 0.089911]]]):
    # Normalize every frame using "mean" and "standard deviation from ARID dataset"
    buffer = (buffer - np.array(mean)) / np.array(std)
    return buffer


def downsample(buffer, start=0, interval=2):
    # downsample the video along the temporal axis with step=interval
    return buffer[start::interval, :, :, :]


def random_sample(buffer, clip_len):
    # get clip_len consecutive frames start from a randomly selected index
    length = len(buffer)
    if clip_len > length:
        raise Exception("---clip_len too large!---")
    start_idx = np.random.randint(0, length - clip_len + 1)
    return buffer[start_idx:start_idx + clip_len, :, :, :]


def center_sample(buffer, clip_len):
    length = len(buffer)
    if clip_len > length:
        raise Exception("---clip_len too large!---")
    center_idx = length // 2
    offset = clip_len // 2
    extra = clip_len % 2
    return buffer[center_idx - offset:center_idx + offset + extra, :, :, :]


def get_time_diff(buffer):
    # get time difference images by subtracting img(t+1) and img(t)
    former = buffer[:-1]
    latter = buffer[1:]
    diff = latter - former
    return diff


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def gamma_correction(buffer, gamma):
    invGamma = 1.0 / gamma
    LU_table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    for i, img in enumerate(buffer):
        buffer[i] = cv2.LUT(img, LU_table)
    return buffer


'''def gamma_correct(buffer):
    img_gray = rgb2gray(buffer)
    for i, img in enumerate(buffer):
        #img_gray = rgb2gray(buffer)
        mean = np.mean(img_gray[i])
        gamma_val = math.log10(0.5) / math.log10(mean / 255.0)
        buffer[i] = np.power(img / 255.0, gamma_val)
    return buffer'''


def enhance(buffer, mode='normalize', gamma=2):
    if mode == 'normalize':
        buffer = buffer / 255.0
        return normalize(buffer)
    elif mode == 'gamma':
        buffer = gamma_correction(buffer.astype('uint8'), gamma).astype('float32')
        buffer = buffer / 255.0
        '''buffer = normalize(buffer,
                        mean=[0.23571073453974717, 0.22415162539268774, 0.22899710358427983],
                        std=[0.11859026215071299, 0.11716281818325791, 0.10664881674945043])'''
        return buffer
    else:
        raise Exception('---No enhancement method named {}---'.format(mode))