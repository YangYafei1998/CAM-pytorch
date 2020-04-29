import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from models.racnn_base import RACNN

'''
RACNN PAPER IMPLEMENTATION
'''
class RACNN_original_apn(RACNN):
    def __init__(self, num_classes, device, out_h=224, out_w=224):
        super(RACNN_original_apn, self).__init__(num_classes, device, out_h=224, out_w=224)

        """
        if you need use origin apn, use this part
        """
        # self.apn_conv_01 = nn.Sequential(
        #     nn.Conv2d(512, 64, kernel_size=1, bias=False),
        #     nn.LeakyReLU(0.2),
        #     )
        self.apn_regress_01 = nn.Sequential(
            nn.Linear(14*14*512, 1000, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 3, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Sigmoid() ## chosen
        ) 

        ## print the network architecture
        print(self)


    def apn_map(self, xconv, lvl):
        if lvl == 0:
            t = self.apn_regress_01(xconv.flatten(start_dim=1))
            t = (t-self.shift)*self.scale
            return t
        else:
            raise NotImplementedError
        
    def forward(self, x, target, train_config=-1):
        
        # alternating between two modes
        if train_config == 1:
            self.freeze_network(self.base)
            self.freeze_network(self.base1)
            # self.freeze_network(self.base2)
            self.freeze_network(self.conv_scale_0)
            self.freeze_network(self.conv_scale_1)
            # self.freeze_network(self.conv_scale_2)
            self.freeze_network(self.classifier_0)
            self.freeze_network(self.classifier_1)
            # self.freeze_network(self.classifier_2)
        else:
            self.unfreeze_network(self.base)
            self.unfreeze_network(self.base1)
            # self.unfreeze_network(self.base2)
            self.unfreeze_network(self.conv_scale_0)
            self.unfreeze_network(self.conv_scale_1)
            # self.unfreeze_network(self.conv_scale_2)
            self.unfreeze_network(self.classifier_0)
            self.unfreeze_network(self.classifier_1)
            # self.unfreeze_network(self.classifier_2)
            # self.unfreeze_network(self.apn_conv_01)
            self.unfreeze_network(self.apn_regress_01)
            # self.unfreeze_network(self.apn_map_flatten_01)

        ### Scale 0
        out_0, f_conv0 = self.classification(x, lvl=0)
        
        ### Scale 1
        # t0 = self.apn_map(f_conv0, lvl=0) ## [B, 3]
        t0 = self.apn_map(f_conv0, lvl=0) ## [B, 3]
        # grid = self.grid_sampler(t0) ## [B, H, W, 2]
        x1 = F.grid_sample(x, self.grid_sampler(t0), align_corners=False, padding_mode='border') ## [B, 3, H, W] sampled using grid parameters
        out_1, f_conv1 = self.classification(x1, lvl=1)


        return [out_0, out_1], [t0]
