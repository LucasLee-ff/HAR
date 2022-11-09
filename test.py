import torch.nn as nn
import torch
from models.net import SlowfastNL, Slowfast
from dataset import DarkVid
from torchvision.transforms import CenterCrop
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm


def test(model, test_loader, criterion):
    model.eval()
    loss_sum, n = 0, 0
    preds, targets = [], []
    prediction = []
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for i, data in pbar:
        n += 1

        slow, fast, target = data
        slow = slow.cuda().float()
        fast = fast.cuda().float()
        target = target.cuda()

        inputs_var = [slow, fast]
        with torch.no_grad():
            output = model(inputs_var)
            preds.append(output.detach())
            prediction.append(output.detach().argmax(dim=1))
            targets.append(target)
            loss = criterion(output, target)
            loss_sum += loss.detach()
        pbar.set_description('Validating')
    loss_avg = loss_sum / n
    preds = torch.cat(preds).cpu()
    targets = torch.cat(targets).cpu()
    top1 = top_k_accuracy_score(targets, preds, k=1)
    top5 = top_k_accuracy_score(targets, preds, k=5)
    return loss_avg, top1, top5, prediction


mapping_table = {0: 'Drink', 1: 'Jump', 2: 'Pick', 3: 'Pour', 4: 'Push', 5: 'Run', 6: 'Sit', 7: 'Stand', 8: 'Turn', 9: 'Walk'}

model_path = '/home/lzf/HAR/ckpts/slowfast_nl/LR0.0010_B16/best_model.pth'
model = SlowfastNL()
model = nn.DataParallel(model)
model = model.cuda()
state_dict = torch.load(model_path, map_location=torch.device('cuda'))['state_dict']
model.load_state_dict(state_dict)

test_transforms = CenterCrop((224, 224))
test_loader = DataLoader(DarkVid('/home/lzf/HAR/data/',
                                      mode='test',
                                      clip_len=32,
                                      transform=test_transforms,
                                      modality='rgb',
                                      enhancement='normalize'),
                              batch_size=32, num_workers=6)

criterion = nn.CrossEntropyLoss().cuda()
model.eval()
test_loss, test_top1, test_top5, prediction = test(model, test_loader, criterion)
print('avg test loss:', test_loss.item())
print('top1 acc:', test_top1)
print('top5 acc:', test_top5)
prediction = torch.cat(prediction)
with open('./output.txt', 'w') as f:
    for i, p in enumerate(prediction):
        f.write(str(i) + '\t' + str(p.item()) + '\n')
