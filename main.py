import torch
import torch.nn as nn
from dataset import DarkVid
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import argparse
from models.net import Net
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, CenterCrop


def main(args):
    torch.manual_seed(2022)
    torch.cuda.manual_seed(2022)
    np.random.seed(2022)

    model_name = 'r3d'
    model = Net(model_name=model_name, num_classes=10)
    if args.pretrained:
        pretrained_path = args.pretrained
    else:
        pretrained_path = './models/pth_from_pytorch/r3d_18-b3b3357e.pth'
    model.load_pretrained(pretrained_path)
    #model.load_pretrained('./models/pth_from_pytorch/r2plus1d_18-91a641e6.pth')
    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 30, 70])

    train_transforms = nn.Sequential(
        RandomHorizontalFlip(),
        RandomCrop((224, 224))
    )
    validation_transforms = CenterCrop((224, 224))
    train_loader = DataLoader(DarkVid('./data', mode='train', transform=train_transforms), batch_size=args.batch, shuffle=True)
    valid_loader = DataLoader(DarkVid('./data', mode='validate', transform=validation_transforms), batch_size=args.batch)

    if args.writer:
        writer_path = args.writer
    else:
        writer_path = './log/'
    if not os.path.isdir(writer_path):
        os.makedirs(writer_path)
    settings = 'LR{:.4f}_B{:d}'.format(args.lr, args.batch)
    writer = SummaryWriter(writer_path + settings)

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

    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        checkpoint = './ckpts/'
    if not os.path.isdir(writer_path):
        os.makedirs(writer_path)

    accumulation_step = args.accumulation_step
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_top1, train_top5 = train(model, train_loader, criterion, optimizer, epoch, accumulation_step)
        valid_loss, valid_top1, valid_top5 = test(model, valid_loader, criterion)
        scheduler.step()

        file_name_last = os.path.join(checkpoint, 'model_epoch_%d.pth' % (epoch + 1,))
        file_name_former = os.path.join(checkpoint, 'model_epoch_%d.pth' % epoch)

        if valid_top1 > best_acc:
            best_acc = valid_top1
            if os.path.isfile(checkpoint + 'best_model.pth'):
                os.remove(checkpoint + 'best_model.pth')
            #torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, checkpoints + 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
                'best_acc': best_acc
            }, checkpoint + 'best_model.pth')

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
    model.train()
    loss_sum, n = 0, 0
    preds, targets = [], []
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (input, target) in pbar:
        n += 1
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        preds.append(output.detach())
        #preds.append(torch.argmax(output, dim=1))
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
    model.eval()
    loss_sum, n = 0, 0
    preds, targets = [], []
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for i, (input, target) in pbar:
        n += 1
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            output = model(input)
            preds.append(output.detach())
            #preds.append(torch.argmax(output, dim=1))
            targets.append(target)
            loss = criterion(output, target)
            loss_sum += loss.detach()
        pbar.set_description('Validating')
    loss_avg = loss_sum / n
    preds = torch.cat(preds).cpu()
    top1 = top_k_accuracy_score(targets, preds, k=1)
    top5 = top_k_accuracy_score(targets, preds, k=5)
    return loss_avg, top1, top5


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to train')
    parser.add_argument("--batch", type=int, default=5,
                        help="batch size")
    parser.add_argument("--lr", default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument("--wd", type=float, default=1e-4,
                        help="weight decay")
    parser.add_argument("--test_interval", type=int, default=1,
                        help="By training how many epochs to conduct testing once")
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--writer', default='./drive/MyDrive/log/', type=str, metavar='PATH',
                        help='path of summarywriter')
    parser.add_argument('--checkpoint', default='./drive/MyDrive/ckpts/', type=str, metavar='PATH',
                        help='path to save model')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--accumulation-step', default=1, type=int, metavar='N',
                        help='')
    parser.add_argument('--pretrained', default='./drive/MyDrive/r3d_18-b3b3357e.pth', type=str, metavar='PATH',
                        help='path of pretrained model')
    args = parser.parse_args()
    main(args)
