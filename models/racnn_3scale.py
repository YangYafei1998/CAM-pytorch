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




class RACNN3Scale(nn.Module):

    def __init__(self, num_classes, lvls, device, out_h=224, out_w=224):
        super(RACNN3Scale, self).__init__()
        
        self.num_classes = num_classes
        
        ## global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        ## dropout
        self.drop = nn.Dropout2d(0.5)
        ## sampler
        self.grid_sampler = GridSampler(device, out_h, out_w)
        
        basemodel = models.resnet50(pretrained=True)
        # basemodel = models.resnet18(pretrained=True)
        self.lvls = lvls
        self.baseList = nn.ModuleList()
        self.convList=nn.ModuleList()
        self.clsfierList=nn.ModuleList()
        for _ in range(self.lvls):
            ## list of backbones
            self.baseList.append(
                nn.Sequential(
                    basemodel.conv1,
                    basemodel.bn1,
                    basemodel.relu,
                    basemodel.maxpool,
                    basemodel.layer1,
                    basemodel.layer2,
                    basemodel.layer3
                )
            )
            ## list of classification heads
            self.convList.append(
                nn.Sequential(
                    nn.Conv2d(1024, 512, kernel_size=1, bias=False), ## resnet50
                    # nn.Conv2d(256, 512, kernel_size=1, bias=False), ## resnet18
                    nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True)
                )
            )
            self.clsfierList.append(nn.Linear(512, num_classes))
            
        ## attention proposal head between scales 0 and 1
        self.apn_conv_01 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2),
            )
        self.apn_regress_01 = nn.Sequential(
            nn.Linear(14*14*64, 1000, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1000, 3, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Sigmoid() ## chosen
        ) 

        ## print the network architecture
        print(self)
    
    def classification(self, x, lvl):
        x = self.baseList[lvl](x)
        f_conv = self.convList[lvl](x)
        f_gap = self.gap(f_conv)
        return self.clsfierList[lvl](self.drop(f_gap).squeeze(2).squeeze(2)), f_gap, f_conv
            
    def apn_map(self, xconv, target, lvl):
        shift = torch.tensor([[0.5, 0.5, -1.0]]).to(xconv.device)
        scale = torch.tensor([[1.0, 1.0, 0.4]]).to(xconv.device)
        if lvl == 0:
            t = self.apn_regress_01(self.apn_conv_01(xconv).flatten(start_dim=1))
            # shift the sigmoid output to ( [-0.5,0.5], [-0.5,0.5], [0.4,0.8] ) 
            t = (t-shift)*scale
            return t
        else:
            raise NotImplementedError
        
    def forward(self, x, target, train_config=-1):
        
        # alternating between two modes
        if train_config == 1:
            self.freeze_network(self.baseList)
            self.freeze_network(self.convList)
            self.freeze_network(self.clsfierList)
        else:
            self.unfreeze_network(self.baseList)
            self.unfreeze_network(self.convList)
            self.unfreeze_network(self.clsfierList)
            self.unfreeze_network(self.apn_conv_01)
            self.unfreeze_network(self.apn_regress_01)


        ## 
        out_list = []
        t_list = []

        ## classification
        out, _, f_conv = self.classification(x, 0)
        out_list.append(out)
        xz = x.clone()
        for lvl in range(1,self.lvls):
            if target is None:
                target = torch.argmax(out, dim=-1).unsqueeze(1) ## [B, 1]
            t = self.apn_map(f_conv, target=None, lvl=0) ## [B, 3]
            t_list.append(t)
            grid = self.grid_sampler(t) ## [B, H, W, 2]
            xz = F.grid_sample(xz, grid, align_corners=False, padding_mode='reflection') ## [B, 3, H, W] sampled using grid parameters
            out, _, f_conv = self.classification(xz, lvl)
            out_list.append(out)

        return out_list, t_list

    def freeze_network(self, module):
        for p in module.parameters():
            p.requires_grad = False

    def unfreeze_network(self, module):
        for p in module.parameters():
            p.requires_grad = True