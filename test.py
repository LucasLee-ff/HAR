import torch
import torch.nn as nn
from dataset import DarkVid
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from models.net import Net


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
            preds.append(torch.argmax(output, dim=1))
            targets.append(target)
            loss = criterion(output, target)
            loss_sum += loss.detach()
        pbar.set_description('Validating')
    loss_avg = loss_sum / n
    preds = torch.cat(preds).cpu()
    targets = torch.cat(targets).cpu()
    acc = accuracy_score(targets, preds)
    return loss_avg, acc


model = Net(num_classes=10)
model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
valid_loader = DataLoader(DarkVid('./data', mode='validate'), batch_size=2)

checkpoint = torch.load('./ckpts/model_epoch_100.tar')
model.load_state_dict(checkpoint['state_dict'])
valid_loss, valid_acc = test(model, valid_loader, criterion)
print(valid_loss, valid_acc)
