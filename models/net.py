from torchvision.models.video import r3d_18, r2plus1d_18
import torch.nn as nn
import torch
import pytorchvideo.models
import pytorchvideo.models.x3d as X3D


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
                                           model_num_class=num_classes)
            self.new_layers = ['blocks.5.proj.weight', 'blocks.5.proj.bias']
        else:
            self.backbone = pytorchvideo.models.create_slowfast(model_num_class=num_classes)
            self.new_layers = ['blocks.6.proj.weight', 'blocks.6.proj.bias']

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
