import torch
import torch.nn as nn
from dataset import DarkVid
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import top_k_accuracy_score
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import argparse
from models.net import Net, MultiScaleNet, SlowfastNL
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, CenterCrop, Resize


isSlowfast = False
isMultiScale = False


def main(args):
    torch.manual_seed(2022)
    torch.cuda.manual_seed(2022)
    np.random.seed(2022)

    model_name = args.model.lower()
    global isSlowfast
    global isMultiScale
    if 'slowfast' in model_name:
        isSlowfast = True
        clip_len = 32
        if 'multiscale' in model_name:
            isMultiScale = True
    else:
        isSlowfast = False
        clip_len = 16

    if isMultiScale:
        model = MultiScaleNet(num_classes=10)
    elif model_name == 'slowfast_nl':
        model = SlowfastNL(num_classes=10)
    else:
        model = Net(model_name=model_name, num_classes=10)
    new_layers = model.new_layers

    if args.pretrained:
        pretrained_path = args.pretrained
        model.load_pretrained(pretrained_path)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    base_params = []
    classifier_params = []

    for param, val in model.named_parameters():
        if param in new_layers:
            classifier_params.append(val)
        else:
            base_params.append(val)
    params = [{'params': base_params, 'lr_mult': 1}, {'params': classifier_params, 'lr_mult': 1}]
    assert args.optim == 'adam' or args.optim == 'sgd', 'no such optimizer option'
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 30, 70])

    train_transforms = nn.Sequential(RandomHorizontalFlip(), RandomCrop((224, 224)))
    validation_transforms = CenterCrop((224, 224))
    multi_scale_transforms = Resize(112) if isMultiScale else None
    train_loader = DataLoader(DarkVid('./data',
                                      mode='train',
                                      clip_len=clip_len,
                                      transform=train_transforms,
                                      multi_scale=multi_scale_transforms),
                              batch_size=args.batch, shuffle=True)
    valid_loader = DataLoader(DarkVid('./data',
                                      mode='validate',
                                      clip_len=clip_len,
                                      transform=validation_transforms,
                                      multi_scale=multi_scale_transforms),
                              batch_size=args.val_batch)

    if args.writer:
        writer_path = args.writer
    else:
        writer_path = './log_' + model_name + '/'
    if not os.path.isdir(writer_path):
        os.makedirs(writer_path)
    settings = 'LR{:.4f}_B{:d}'.format(args.lr, args.batch * args.accumulation_step)
    writer = SummaryWriter(os.path.join(writer_path, settings))

    if args.resume and os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_dict'])
        best_acc = checkpoint['best_acc']
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        best_acc = 0
        print("=> no checkpoint found")

    if args.checkpoint_path:
        checkpoint_path = os.path.join(args.checkpoint_path, settings)
    else:
        checkpoint_path = './ckpts/' + model_name
        checkpoint_path = os.path.join(checkpoint_path, settings)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    accumulation_step = args.accumulation_step
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_top1, train_top5 = train(model, train_loader, criterion, optimizer, epoch, accumulation_step)
        valid_loss, valid_top1, valid_top5 = test(model, valid_loader, criterion)
        scheduler.step()

        file_name_last = os.path.join(checkpoint_path, 'model_epoch_%d.pth' % (epoch + 1,))
        file_name_former = os.path.join(checkpoint_path, 'model_epoch_%d.pth' % epoch)

        if valid_top1 >= best_acc:
            best_acc = valid_top1
            best_model_path = os.path.join(checkpoint_path, 'best_model.pth')
            if os.path.isfile(best_model_path):
                os.remove(best_model_path)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
                'best_acc': best_acc
            }, best_model_path)

        # save the latest model
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'scheduler_dict': scheduler.state_dict(),
            'best_acc': best_acc
        }, file_name_last)

        # delete models from former epochs
        if epoch != 0:
            os.remove(file_name_former)

        writer.add_scalars('Loss', {'train': train_loss, 'validation': valid_loss}, epoch + 1)
        writer.add_scalars('Top1', {'train': train_top1, 'validation': valid_top1}, epoch + 1)
        writer.add_scalars('Top5', {'train': train_top5, 'validation': valid_top5}, epoch + 1)


def train(model, train_loader, criterion, optimizer, epoch, accumulation_steps=1):
    global isSlowfast
    global isMultiScale
    model.train()
    loss_sum, n = 0, 0
    preds, targets = [], []
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, data in pbar:
        n += 1
        if isMultiScale:
            inputs1, inputs2, target = data
            inputs1 = inputs1.cuda()
            inputs2 = inputs2.cuda()
            target = target.cuda()
            inputs_var = [[inputs1[:, :, ::4, :, :], inputs1], [inputs2[:, :, ::4, :, :], inputs2]]
        else:
            inputs, target = data
            inputs = inputs.cuda()
            target = target.cuda()
            if isSlowfast:
                inputs_var = [inputs[:, :, ::4, :, :], inputs]
            else:
                inputs_var = inputs
        output = model(inputs_var)
        preds.append(output.detach())
        targets.append(target)

        loss = criterion(output, target)
        loss_sum += loss.detach()
        loss = loss / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        pbar.set_description('Epoch {:03d}'.format(epoch + 1))
    loss_avg = loss_sum / n
    preds = torch.cat(preds).cpu()
    targets = torch.cat(targets).cpu()
    top1 = top_k_accuracy_score(targets, preds, k=1)
    top5 = top_k_accuracy_score(targets, preds, k=5)
    return loss_avg, top1, top5


def test(model, test_loader, criterion):
    global isSlowfast
    global isMultiScale
    model.eval()
    loss_sum, n = 0, 0
    preds, targets = [], []
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for i, data in pbar:
        n += 1
        if isMultiScale:
            inputs1, inputs2, target = data
            inputs1 = inputs1.cuda()
            inputs2 = inputs2.cuda()
            target = target.cuda()
            inputs_var = [[inputs1[:, :, ::4, :, :], inputs1], [inputs2[:, :, ::4, :, :], inputs2]]
        else:
            inputs, target = data
            inputs = inputs.cuda()
            target = target.cuda()
            if isSlowfast:
                inputs_var = [inputs[:, :, ::4, :, :], inputs]
            else:
                inputs_var = inputs
        with torch.no_grad():
            output = model(inputs_var)
            preds.append(output.detach())
            targets.append(target)
            loss = criterion(output, target)
            loss_sum += loss.detach()
        pbar.set_description('Validating')
    loss_avg = loss_sum / n
    preds = torch.cat(preds).cpu()
    targets = torch.cat(targets).cpu()
    top1 = top_k_accuracy_score(targets, preds, k=1)
    top5 = top_k_accuracy_score(targets, preds, k=5)
    return loss_avg, top1, top5


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='r3d', type=str,
                        help='model to use, can be slowfast_nl, multiscale_slowfast, slowfast, r3d, r2+1d, x3d')

    # basic
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to train')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument("--batch", type=int, default=16,
                        help="batch size")
    parser.add_argument("--val-batch", type=int, default=32,
                        help="batch size for validation set")
    # optimizer
    parser.add_argument('--optim', default='adam', type=str,
                        help='optimizer')
    parser.add_argument("--lr", default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument("--wd", type=float, default=1e-4,
                        help="weight decay")
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--accumulation-step', default=1, type=int,
                        help='number of batch to calculate before update net params')

    # path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--writer', default='', type=str,
                        help='path of SummaryWriter')
    parser.add_argument('--checkpoint-path', default='', type=str,
                        help='path to save model')
    parser.add_argument('--pretrained', default='', type=str,
                        help='path of pretrained model')
    args = parser.parse_args()
    main(args)
