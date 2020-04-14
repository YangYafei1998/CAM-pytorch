import torch
import torch.nn as nn
import torch.nn.functional as F


class RACNN(nn.Module):
    def __init__(self):
        super(RACNN, self).__init__()

        basemodel = models.resnet50(pretrained=pretrained)
        
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

        ## final conv for scale 0
        self.conv_scale_0 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            )

        ## final conv for scale 1
        self.conv_scale_1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            )

        # ## final conv for scale 2
        # self.conv_scale_2 = nn.Sequential(
        #     nn.Conv2d(1024, 512, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     )

        ## global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        ## dropout
        self.drop = nn.Dropout2d(0.5)

        ## attention proposal head between scales 0 and 1
        self.apn_scale_01 = nn.Sequential(
            nn.Linear(512, 3, bias=False)
            nn.Sigmoid()
        )

        # ## attention proposal head between scales 1 and 2
        # self.apn_scale_12 = nn.Sequential(
        #     nn.Linear(512, 3, bias=False)
        #     nn.Sigmoid()
        # )

        ## classification head
        self.classifier = nn.Linear(512, num_classes)

        ## print the network architecture
        print(self)
    
    def classification(self, x, lvl):
        f = self.base(x)
        if lvl == 0:
            f = self.conv_scale_0(x)
            x = self.gap(f)
        elif lvl == 1:
            f = self.conv_scale_1(x)
            x = self.gap(f)
        else:
            raise NotImplementedError

        x = self.drop(x)
        x = self.classifier(x.squeeze(2).squeeze(2))
        return x

    def apn(self, x, lvl, detach=False):
        f = self.base(x) 
        ## only train APN (can be used in pretrain APN)
        if detach:
            f = f.detach()
        if lvl == 0:
            f = self.drop(f)
            t = self.apn_scale_01(f.squeeze(2).squeeze(2))
        else:
            raise NotImplementedError

    def forward(self, x, lvl, apn):
        if apn:
            return self.apn(x, lvl)
        else:
            return self.classification(x, lvl)
