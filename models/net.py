from torchvision.models.video import r3d_18, r2plus1d_18
import torch.nn as nn
import torch
import pytorchvideo.models


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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['fc.weight', 'fc.bias']}
        model_dict = self.backbone.state_dict()
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)


class SlowFast(nn.Module):
    def __init__(self, depth=50, num_classes=10):
        super(SlowFast, self).__init__()
        assert depth == 50, "Only support resnet50 currently"
        self.backbone = pytorchvideo.models.create_slowfast(model_depth=depth, model_num_class=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x

    def load_pretrained(self, path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['blocks.6.proj.weight', 'blocks.6.proj.bias']}
        model_dict = self.backbone.state_dict()
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
