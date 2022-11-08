#The effcientnet_pytorch package is licensed under LGPL V3
#License can be found in the subdir "LICENSES"
from efficientnet_pytorch import EfficientNet

import torch.nn as nn

class new_fc(nn.Module):
    def __init__(self):
        super(new_fc, self).__init__()
        self.linear1_1 = nn.Linear(2560, 512)
        self.linear1_2 = nn.Linear(in_features=512, out_features=128)
        self.linear1_3 = nn.Linear(in_features=128, out_features=128)
        self.linear1_4 = nn.Linear(in_features=128, out_features=2)
        #self.linear1_4 = nn.Linear(in_features=2560, out_features=2)
        self.linear2 = nn.Linear(in_features=2560, out_features=1, bias=True)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        #self.memswish = MemoryEfficientSwish()
        
    def forward(self, x):
        x1 = self.dropout(x)
        x1 = self.linear2(x1)
        
        x2 = self.linear1_1(x)
        x2 = self.linear1_2(x2)
        x2 = self.linear1_3(x2)
        x2 = self.linear1_4(x2)
        
        return x1, x2



class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x    


def effnet(cfg):
    model = EfficientNet.from_pretrained(cfg.NAME, num_classes=cfg.NUM_CLASSES)
    model._fc = new_fc()
    model._dropout = Identity()
    return model
    
    
    
