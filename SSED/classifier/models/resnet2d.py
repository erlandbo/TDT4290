import torch.nn as nn
from torchvision.models import resnet50, resnet34
from collections.abc import Iterable

class ResNet50BB(nn.Module):
    """
    Resnet50 classifier class
    """
    def __init__(self, cfg):
        super(ResNet50BB, self).__init__()
        if getattr(cfg, "PRETRAINED", True) == False:
            self.pretrained = False
        else:
            self.pretrained = True
        self.res = resnet50(pretrained=self.pretrained)
        self.fc = nn.Sequential(
            nn.Linear(in_features=1000, out_features=cfg.NUM_CLASSES)
        )
    def forward(self, x):
        x = self.res(x)
        x = self.fc(x)
        return x



class ResNet34BB(nn.Module):
    """
    Resnet34 classifier class
    """
    def __init__(self, cfg):
        super(ResNet34BB, self).__init__()
        if getattr(cfg, "PRETRAINED", True) == False:
            self.pretrained = False
        else:
            self.pretrained = True
        self.res = resnet34(pretrained=self.pretrained)
        self.fc = nn.Sequential(
            nn.Linear(in_features=1000, out_features=cfg.NUM_CLASSES)
        )
    def forward(self, x):
        x = self.res(x)
        x = self.fc(x)
        return x
