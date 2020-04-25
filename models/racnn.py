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

    def __init__(self, num_classes, device, out_h=224, out_w=224):
        super(RACNN, self).__init__()
        
        # shift the sigmoid output to ( [-0.5,0.5], [-0.5,0.5], [0.4,0.8] ) 
        self.shift = torch.tensor([[0.5, 0.5, -1.0]]).to(device)
        self.scale = torch.tensor([[1.0, 1.0, 0.4]]).to(device)

        self.num_classes = num_classes
        basemodel = models.resnet50(pretrained=True)
        # basemodel = models.resnet18(pretrained=True)
        
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
        self.base1 = nn.Sequential(
            basemodel.conv1,
            basemodel.bn1,
            basemodel.relu,
            basemodel.maxpool,
            basemodel.layer1,
            basemodel.layer2,
            basemodel.layer3
        )
        self.base2 = nn.Sequential(
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
            nn.Conv2d(1024, 512, kernel_size=1, bias=False), ## resnet50
            # nn.Conv2d(256, 512, kernel_size=1, bias=False), ## resnet18
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
            )
        self.classifier_0 = nn.Linear(512, num_classes)

        ## classification head for scale 1
        self.conv_scale_1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False), ## resnet50
            # nn.Conv2d(256, 512, kernel_size=1, bias=False), ## resnet18
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
            )
        self.classifier_1 = nn.Linear(512, num_classes)

        ## classification head for scale 2
        self.conv_scale_2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False), ## resnet50
            # nn.Conv2d(256, 512, kernel_size=1, bias=False), ## resnet18
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
            )
        self.classifier_2 = nn.Linear(512, num_classes)

        ## final conv for scale 2
        self.conv_scale_2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False), ## resnet50
            # nn.Conv2d(256, 512, kernel_size=1, bias=False), ## resnet18
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
            )

        ## attention proposal head between scales 0 and 1
        self.apn_conv_01 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2),
            )
        self.apn_regress_01 = nn.Sequential(
            nn.Linear(14*14*64, 1000, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 3, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Sigmoid() ## chosen
        ) 
        self.apn_map_flatten_01 = nn.Sequential(
            nn.Linear(14*14, 3, bias=True),
            nn.Sigmoid()
        )

        ## print the network architecture
        print(self)
    
    def classification(self, x, lvl):
        # x = self.base(x)
        if lvl == 0:
            x = self.base(x)
            f_conv = self.conv_scale_0(x)
            f_gap = self.gap(f_conv)
            return self.classifier_0(self.drop(f_gap).squeeze(2).squeeze(2)), f_gap, f_conv
        elif lvl == 1:
            x = self.base1(x)
            f_conv = self.conv_scale_1(x)
            f_gap = self.gap(f_conv)
            return self.classifier_1(self.drop(f_gap).squeeze(2).squeeze(2)), f_gap, f_conv
        else:
            raise NotImplementedError
            
    def apn_map(self, xconv, lvl):
        if lvl == 0:
            t = self.apn_regress_01(self.apn_conv_01(xconv).flatten(start_dim=1))
            t = (t-self.shift)*self.scale
            return t
        else:
            raise NotImplementedError

    def apn_map_chlwise(self, xconv, lvl):
        if lvl == 0:
            ## channelwise pooling 
            B, C = xconv.shape[0:2]
            xconv = xconv.view(B, C, -1)
            xconv = xconv.permute(0,2,1)
            xconv = F.adaptive_avg_pool1d(xconv, 1).squeeze(2)
            t = self.apn_map_flatten_01(xconv.flatten(start_dim=1))
            t = (t-self.shift)*self.scale
            return t
        else:
            raise NotImplementedError
        
    def forward(self, x, target, train_config=-1):
        
        # alternating between two modes
        if train_config == 1:
            self.freeze_network(self.base)
            self.freeze_network(self.base1)
            self.freeze_network(self.base2)
            self.freeze_network(self.conv_scale_0)
            self.freeze_network(self.conv_scale_1)
            self.freeze_network(self.conv_scale_2)
            self.freeze_network(self.classifier_0)
            self.freeze_network(self.classifier_1)
            self.freeze_network(self.classifier_2)
        else:
            self.unfreeze_network(self.base)
            self.unfreeze_network(self.base1)
            self.unfreeze_network(self.base2)
            self.unfreeze_network(self.conv_scale_0)
            self.unfreeze_network(self.conv_scale_1)
            self.unfreeze_network(self.conv_scale_2)
            self.unfreeze_network(self.classifier_0)
            self.unfreeze_network(self.classifier_1)
            self.unfreeze_network(self.classifier_2)
            self.unfreeze_network(self.apn_conv_01)
            self.unfreeze_network(self.apn_regress_01)


        ## classification scale 1
        out_0, f_gap_0, f_conv0 = self.classification(x, lvl=0)
        
        # t0 = self.apn_map(f_conv0, lvl=0) ## [B, 3]
        t0 = self.apn_map_chlwise(f_conv0, lvl=0) ## [B, 3]
        
        grid = self.grid_sampler(t0) ## [B, H, W, 2]
        x1 = F.grid_sample(x, grid, align_corners=False, padding_mode='border') ## [B, 3, H, W] sampled using grid parameters

        ## classification scale 2
        out_1, f_gap_1, f_conv1 = self.classification(x1, lvl=1)

        return out_0, out_1, t0, f_gap_1

    def freeze_network(self, module):
        for p in module.parameters():
            p.requires_grad = False

    def unfreeze_network(self, module):
        for p in module.parameters():
            p.requires_grad = True