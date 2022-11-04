import torch
from saliency_detection.model.csnet import build_model


def get_fast_input(model, buffer, diff):
    # buffer: (B, 3, T, H, W)
    # diff: (B, T, H, W, 3)
    new_diff = torch.empty_like(diff, dtype=torch.float32)
    diff = diff.cuda()
    buffer = buffer.permute((0, 2, 1, 3, 4))  # (B, 3, T, H, W) -> (B, T, 3, H, W)
    for i in range(len(buffer)):
        saliency_maps = get_saliency(model, buffer[i])  # (T, 3, H, W) -> (T, 1, H, W)
        saliency_maps = saliency_maps.permute((0, 2, 3, 1))  # (T, 1, H, W) -> (T, H, W, 1)
        saliency_inter = get_saliency_intersection(saliency_maps)  # (T, H, W, 1)
        saliency_inter = saliency_inter.repeat(1, 1, 1, 3)  # (T, H, W, 1) -> (T, H, W, 3)
        new_diff[i] = mask_with_roi(diff[i], saliency_inter)  # (T, H, W, 3)
    new_diff = new_diff.permute((0, 4, 1, 2, 3))  # (B, 3, T, H, W)
    return new_diff


def bulid_saliency_model():
    model = build_model(predefine="/home/lzf/HAR/saliency_detection/checkpoints/csnet-L-x2/csnet-L-x2.bin")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load("/home/lzf/HAR/saliency_detection/checkpoints/csnet-L-x2/csnet-L-x2.pth.tar", map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def get_saliency(model, buffer):
    # buffer: (B, 3, 224, 224)
    with torch.no_grad():
        predict = model(buffer.float())
        predict = torch.sigmoid(predict)  # (B, 1, 224, 224)
    return predict


def get_saliency_batch(model, batch):
    # batch: (B, C, T, H, W)
    batch = batch.permute((0, 2, 1, 3, 4))  # (B, T, C, H, W)
    B, T, C, H, W = batch.shape
    new_batch = torch.empty((B, T, C+1, H, W))
    for i in range(len(batch)):
        saliency = get_saliency(model, batch[i])  # (T, 1, H, W)
        new_batch[i] = torch.cat((batch[i], saliency), dim=1)  # (T, C+1, H, W)
    new_batch = new_batch.permute((0, 2, 1, 3, 4))
    return new_batch


def get_saliency_intersection(saliency_maps):
    # saliency_maps: (B, 1, 224, 224)
    former = saliency_maps[:-1]
    latter = saliency_maps[1:]
    intersection = torch.where(latter > former, latter, former)
    intersection = torch.where(intersection >= 0.5, 1, 0)
    return intersection


def mask_with_roi(imgs, roi):
    result = imgs * roi
    '''for i, frame in enumerate(result):
        result[i] = (frame - frame.min()) / (frame.max() - frame.min())'''
    return result
