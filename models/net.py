from torchvision.models.video import r3d_18, r2plus1d_18
import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self, model_name='r3d', num_classes=10):
        super(Net, self).__init__()
        assert model_name == 'r3d' or model_name == 'r2+1d', "No such model"
        if model_name == 'r3d':
            self.backbone = r3d_18(num_classes=num_classes)
        elif model_name == 'r2+1d':
            self.backbone = r2plus1d_18(num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x

    def load_pretrained(self, path):
        device = torch.device('cuda')
        pretrained_dict = torch.load(path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['fc.weight', 'fc.bias']}
        model_dict = self.backbone.state_dict()
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
