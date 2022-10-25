import torch.nn as nn
import torch
import pytorchvideo.models.x3d
import torch
from models.net import Net, MultiScaleNet
from dataset import DarkVid
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, CenterCrop, Resize
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm


isSlowfast = True
isMultiScale = False


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


mapping_table = {0: 'Drink', 1: 'Jump', 2: 'Pick', 3: 'Pour', 4: 'Push', 5: 'Run', 6: 'Sit', 7: 'Stand', 8: 'Turn', 9: 'Walk'}

model_path = 'D:/CCA/MSc courses/Machine Vision/CA/HAR/ckpts/slowfast/LR0.0010_B16/best_model.pth'
model = Net(model_name='slowfast')
state_dict = torch.load(model_path, map_location=torch.device('cpu'))['state_dict']
model.load_state_dict(state_dict)
train_transforms = nn.Sequential(RandomHorizontalFlip(), RandomCrop((224, 224)))
validation_transforms = CenterCrop((224, 224))
data = DarkVid('D:/CCA/MSc courses/Machine Vision/CA/HAR/data/', mode='validate', clip_len=32, transform=validation_transforms)
loader = DataLoader(data, batch_size=1)
criterion = nn.CrossEntropyLoss()
model.eval()
valid_loss, valid_top1, valid_top5 = test(model, loader, criterion)
print(valid_loss, valid_top1, valid_top5)
