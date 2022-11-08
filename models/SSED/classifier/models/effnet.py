#The effcientnet_pytorch package is licensed under LGPL V3
#License can be found in the subdir "LICENSES"
from efficientnet_pytorch import EfficientNet


def effnet(cfg):
    return EfficientNet.from_pretrained(cfg.NAME, num_classes=cfg.NUM_CLASSES)
