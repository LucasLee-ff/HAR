import os
import importlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io
from skimage.transform import resize
from model.utils.simplesum_octconv import simplesum
import cv2
from dataset_utils import load_video, get_time_diff, downsample, normalize


def main():
    global cfg
    model_lib = importlib.import_module("model." + "csnet")
    predefine_file = "checkpoints/csnet-L-x2/csnet-L-x2.bin"
    model = model_lib.build_model(predefine=predefine_file)
    #model.cuda()
    prams, flops = simplesum(model, inputsize=(3, 224, 224), device=0)
    print('  + Number of params: %.4fM' % (prams / 1e6))
    print('  + Number of FLOPs: %.4fG' % (flops / 1e9))
    this_checkpoint ="checkpoints/csnet-L-x2/csnet-L-x2.pth.tar"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.isfile(this_checkpoint):
        print("=> loading checkpoint '{}'".format(this_checkpoint))
        checkpoint = torch.load(this_checkpoint, map_location=device)
        loadepoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(
            this_checkpoint, checkpoint['epoch']))
    else:
        print(this_checkpoint, "Not found.")

    path = 'D:/CCA/MSc courses/Machine Vision/CA/HAR/data/train/Drink/Drink_3_1.mp4'
    buffer = load_video(path)
    buffer = buffer / 255.0
    buffer = normalize(buffer)
    buffer = downsample(buffer)
    diff = get_time_diff(buffer)
    curr = None
    former = None
    '''
    for i, img in enumerate(buffer):
        output = test_no_norm(model, img)
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        output = output.astype('float32')
        img_224 = cv2.resize(img, (224, 224))
        result = cv2.bitwise_and(img_224, output)
        result *= 255
        cv2.imwrite('./test/' + str(i) + '.jpg', result)
    '''

    for i, img in enumerate(buffer):
        former = curr
        curr = test_no_norm(model, img)
        curr = cv2.cvtColor(curr, cv2.COLOR_GRAY2BGR)
        curr = curr.astype('float32')
        if former is not None:
            #roi = cv2.bitwise_and(curr, former)
            roi = np.where(curr > former, curr, former)
            diff_224 = cv2.resize(diff[i-1], (224, 224))
            #result = cv2.bitwise_and(roi, diff_224)
            result = roi * diff_224
            result = (result - result.min()) / (result.max() - result.min())
            result *= 255
            cv2.imwrite('./test/' + str(i) + '.jpg', result)

    #img=io.imread("00000.jpg")
    #output=test_no_norm(model,img) #input not norm

    #plt.imshow(output)
    #plt.show()
    # output=test_have_norm(model,img)#


def test_no_norm(model, img):
    print("Start testing.")
    mean = [0.07076896,0.06474562,0.06435021]
    std = [0.06226239,0.05933105,0.05387579]
    with torch.no_grad():
        h, w = img.shape[:2]

        img = resize(img, (224,224),
                     mode='reflect',
                     anti_aliasing=False)
        h, w = img.shape[:2]
        img = np.transpose((img - mean) / std, (2, 0, 1))
        img = torch.unsqueeze(torch.FloatTensor(img), 0)
        input_var = torch.autograd.Variable(img)
        #input_var = input_var.cuda()
        predict = model(input_var)
        predict = predict[0]
        predict = torch.sigmoid(predict.squeeze(0).squeeze(0))
        predict = predict.data.cpu().numpy()
        predict = (resize(
            predict, (h, w), mode='reflect', anti_aliasing=False) *
                   255).astype(np.uint8)
        return predict


def test_have_norm(model, img):

    print("Start testing.")
    with torch.no_grad():
        h, w = img.shape[:2]
        if h !=224 or w!=224 :
            img = resize(img, (224,224),
                         mode='reflect',
                         anti_aliasing=False)
        h, w = img.shape[:2]
        img = torch.unsqueeze(torch.FloatTensor(img), 0)
        input_var = torch.autograd.Variable(img)
        #input_var = input_var.cuda()
        predict = model(input_var)
        predict = predict[0]
        predict = torch.sigmoid(predict.squeeze(0).squeeze(0))
        predict = predict.data.cpu().numpy()
        predict = (resize(
            predict, (h, w), mode='reflect', anti_aliasing=False) *
                   255).astype(np.uint8)
        return predict


if __name__ == '__main__':
    main()
