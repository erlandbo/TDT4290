from classifier.models.resnet2d import ResNet50BB, ResNet34BB
from classifier.models.effnet import effnet

def build_model(cfg):
    if cfg.MODEL.NAME == "resnet50":
        return ResNet50BB(cfg.MODEL)
    elif cfg.MODEL.NAME == "resnet34":
        return ResNet34BB(cfg.MODEL)
    elif "efficientnet" in cfg.MODEL.NAME:
        return effnet(cfg.MODEL)
    else:
        return None
