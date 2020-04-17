import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class GridSampler(nn.Module):
    def __init__(self, device, out_h=224, out_w=224):
        super(GridSampler, self).__init__()
        self.out_w, self.out_h = out_w, out_h
        
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, self.out_w), np.linspace(-1, 1, self.out_h))
        
        self.grid_X = torch.from_numpy(self.grid_X).type(torch.FloatTensor)
        self.grid_X = self.grid_X.unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.from_numpy(self.grid_Y).type(torch.FloatTensor)
        self.grid_Y = self.grid_Y.unsqueeze(0).unsqueeze(3)
        
        self.grid_X = self.grid_X.to(device).requires_grad_(False)
        self.grid_Y = self.grid_Y.to(device).requires_grad_(False)

    def forward(self, theta):
        ## theta \in [0, 1]
        B = theta.shape[0] ## batch size
        # print(theta.shape)
        theta = theta.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        trans_x, trans_y, uni_scale = theta[:, 0, ...], theta[:, 1, ...], theta[:, 2, ...]
        # print(trans_x.shape)

        X = self.grid_X.repeat_interleave(B, dim=0)
        Y = self.grid_Y.repeat_interleave(B, dim=0)
        X = (X + trans_x)*uni_scale
        Y = (Y + trans_y)*uni_scale
        # print(X.shape)
        return torch.cat((X, Y), dim=-1)

class RACNN(nn.Module):

    detach_modules = {'base', 'classifier0', 'classifier1', 'apn01'}

    def __init__(self, num_classes, device, out_h=224, out_w=224):
        super(RACNN, self).__init__()

        basemodel = models.resnet50(pretrained=True)
        
        ## shared feature extractor
        self.base = nn.Sequential(
            basemodel.conv1,
            basemodel.bn1,
            basemodel.relu,
            basemodel.maxpool,
            basemodel.layer1,
            basemodel.layer2,
            basemodel.layer3
        )

        ## global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        ## dropout
        self.drop = nn.Dropout2d(0.5)
        ## sampler
        self.grid_sampler = GridSampler(device, out_h, out_w)

        ## classification head for scale 0
        self.conv_scale_0 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
            )
        self.classifier_0 = nn.Linear(512, num_classes)

        ## classification head for scale 1
        self.conv_scale_1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
            )
        self.classifier_1 = nn.Linear(512, num_classes)

        # ## final conv for scale 2
        # self.conv_scale_2 = nn.Sequential(
        #     nn.Conv2d(1024, 512, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     )

        ## attention proposal head between scales 0 and 1
        self.apn_scale_01 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.Dropout(0.2),
            nn.Linear(256, 128, bias=False),
            nn.Dropout(0.2),
            nn.Linear(128, 3, bias=False),
            nn.Sigmoid()
            )

        ## attention proposal head between scales 0 and 1
        self.apn_map_scale_01 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, bias=False),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 128, kernel_size=3, bias=False),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 3, kernel_size=1, bias=False),
            nn.Sigmoid()
            )

        ## print the network architecture
        print(self)
    
    def classification(self, x, lvl):
        x = self.base(x)
        if lvl == 0:
            f_conv = self.conv_scale_0(x)
            f_gap = self.gap(f_conv)
            return self.classifier_0(self.drop(f_gap).squeeze(2).squeeze(2)), f_gap, f_conv
        elif lvl == 1:
            f_conv = self.conv_scale_1(x)
            f_gap = self.gap(f_conv)
            return self.classifier_1(self.drop(f_gap).squeeze(2).squeeze(2)), f_gap, f_conv
        else:
            raise NotImplementedError
        
    def apn(self, x, lvl):
        # x = self.drop(x)
        # print(x.shape)
        shift = torch.tensor([[0.5,0.5,-0.5]]).to(x.device)
        scale = torch.tensor([[1.0,1.0,0.6]]).to(x.device)
        if lvl == 0:
            # t = self.apn_scale_01(x.squeeze(2).squeeze(2))
            t = self.apn_map_scale_01(x).squeeze(2).squeeze(2)
            # shift the sigmoid output to ( [-0.5,0.5], [-0.5,0.5], [0,0.7] ) 
            t = (t-shift)*scale
            return t
        else:
            raise NotImplementedError
        
    def forward(self, x, train_config=-1):
        
        # alternating between two modes
        if train_config == 1:
            self.freeze_network(self.conv_scale_0)
            self.freeze_network(self.conv_scale_1)
            self.freeze_network(self.classifier_0)
            self.freeze_network(self.classifier_1)
        elif train_config == 0:
            self.freeze_network(self.apn_scale_01)
            
        ## classification scale 1
        out_0, f_gap_0, f_conv0 = self.classification(x, lvl=0)
        ## zoom in
        # t0 = self.apn(f_gap_0, lvl=0) ## [B, 3]
        t0 = self.apn(f_conv0, lvl=0) ## [B, 3]
        grid = self.grid_sampler(t0) ## [B, H, W, 2]
        x1 = F.grid_sample(x, grid, align_corners=False) ## [B, 3, H, W] sampled using grid parameters
        ## classification scale 2
        out_1, f_gap_1, f_conv1 = self.classification(x1, lvl=1)

        return out_0, out_1, t0

    def freeze_network(self, module):
        for p in module.parameters():
            p.requires_grad = False