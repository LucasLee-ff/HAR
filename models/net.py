from torchvision.models.video import r3d_18, r2plus1d_18
import torch.nn as nn
import torch
import pytorchvideo.models
import pytorchvideo.models.x3d as X3D
from .non_local import create_slowfast_nl


class Net(nn.Module):
    def __init__(self, model_name='r3d', num_classes=10):
        super(Net, self).__init__()
        assert model_name in ['r3d', 'r2+1d', 'x3d', 'slowfast'], "No such model"
        self.model_name = model_name
        if self.model_name == 'r3d':
            self.backbone = r3d_18(num_classes=num_classes)
            self.new_layers = ['fc.weight', 'fc.bias']
        elif self.model_name == 'r2+1d':
            self.backbone = r2plus1d_18(num_classes=num_classes)
            self.new_layers = ['fc.weight', 'fc.bias']
        elif self.model_name == 'x3d':
            self.backbone = X3D.create_x3d(input_clip_length=16, input_crop_size=224, depth_factor=2.2,
                                           model_num_class=num_classes, head_activation=None)
            self.new_layers = ['blocks.5.proj.weight', 'blocks.5.proj.bias']

    def forward(self, x):
        x = self.backbone(x)
        return x

    def load_pretrained(self, path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in self.new_layers}
        model_dict = self.backbone.state_dict()
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        print("=> loaded pretrained model {}".format(path))


class MultiScaleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MultiScaleNet, self).__init__()
        self.branch_large = pytorchvideo.models.create_slowfast(model_num_class=num_classes,
                                                                head_pool_kernel_sizes=((8, 7, 7), (32, 7, 7)))
        self.branch_small = pytorchvideo.models.create_slowfast(model_num_class=num_classes,
                                                                head_pool_kernel_sizes=((8, 4, 4), (32, 4, 4)))
        self.new_layers = ['blocks.6.proj.weight', 'blocks.6.proj.bias',
                           'branch_large.blocks.6.proj.weight', 'branch_large.blocks.6.proj.bias',
                           'branch_small.blocks.6.proj.weight', 'branch_small.blocks.6.proj.bias']

    def forward(self, x):
        y1 = self.branch_large(x[0])
        y2 = self.branch_small(x[1])

        # fuse two prediction by averaging
        y3 = torch.stack((y1, y2))
        y = torch.mean(y3, dim=0)
        return y

    def load_pretrained(self, path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in self.new_layers}

        model_dict1 = self.branch_large.state_dict()
        model_dict1.update(pretrained_dict)
        self.branch_large.load_state_dict(model_dict1)

        model_dict2 = self.branch_small.state_dict()
        model_dict2.update(pretrained_dict)
        self.branch_small.load_state_dict(model_dict2)
        print("=> loaded pretrained model {}".format(path))


class Slowfast(nn.Module):
    def __init__(self, num_classes=10, input_channels=(3, 3)):
        super(Slowfast, self).__init__()
        self.backbone = pytorchvideo.models.create_slowfast(model_num_class=num_classes, input_channels=input_channels)
        self.new_layers = ['blocks.6.proj.weight', 'blocks.6.proj.bias']
        if input_channels != (3, 3):
            if input_channels[0] != 3:
                self.new_layers = self.new_layers + ['blocks.0.multipathway_blocks.0.conv.weight']
            if input_channels[1] != 3:
                self.new_layers = self.new_layers + ['blocks.0.multipathway_blocks.1.conv.weight']

    def forward(self, x):
        x = self.backbone(x)
        return x

    def load_pretrained(self, path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in self.new_layers}
        model_dict = self.backbone.state_dict()
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        print("=> loaded pretrained model {}".format(path))


class SlowfastNL(nn.Module):
    def __init__(self, num_classes=10, head_activation=None):
        super(SlowfastNL, self).__init__()
        self.backbone = create_slowfast_nl(model_num_class=num_classes, head_activation=head_activation)
        self.new_layers = ['blocks.6.proj.weight', 'blocks.6.proj.bias']

    def forward(self, x):
        x = self.backbone(x)
        return x

    def load_pretrained(self, path):
        # load kinetics400 pretrained slowfast
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(path, map_location=device)
        new_pretrained_dict = dict()
        for k, v in pretrained_dict.items():
            if k in self.new_layers:
                continue
            param_name = k.split('.')
            n = int(param_name[1])
            if n >= 3:
                param_name[1] = str(n + 1)
                new_pretrained_dict['.'.join(param_name)] = v
            else:
                new_pretrained_dict[k] = v
        model_dict = self.backbone.state_dict()
        model_dict.update(new_pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        print("=> loaded pretrained model {}".format(path))

    def load_pretrained_arid(self, path):
        # load arid pretrained slowfast
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(path, map_location=device)['state_dict']
        new_pretrained_dict = dict()
        for k, v in pretrained_dict.items():
            param_name = k.split('.')
            new_pretrained_dict['.'.join(param_name[1:])] = v
        model_dict = self.state_dict()
        model_dict.update(new_pretrained_dict)
        self.load_state_dict(model_dict)
        print("=> loaded pretrained model {}".format(path))


class MultiBranchSlowfast(nn.Module):
    def __init__(self, num_classes=10):
        super(MultiBranchSlowfast, self).__init__()
        # self.backbone = pytorchvideo.models.create_slowfast(model_num_class=num_classes, head_activation=nn.ReLU)
        self.backbone = SlowfastNL(head_activation=nn.ReLU)
        self.RGB_diff = Net(model_name='x3d', num_classes=num_classes)
        self.fuse = nn.Linear(20, 10)
        self.dropout = nn.Dropout()
        self.new_layers = ['blocks.6.proj.weight', 'blocks.6.proj.bias']

    def forward(self, x):
        y1 = self.backbone(x[:2])
        y2 = self.RGB_diff(x[-1])
        y3 = torch.concat((y1, y2), dim=1)
        y = self.fuse(y3)
        return y

    def load_pretrained(self, path):
        self.RGB_diff.load_pretrained('/home/lzf/HAR/models/X3D_M.pth')
        self.backbone.load_pretrained_arid(path)
