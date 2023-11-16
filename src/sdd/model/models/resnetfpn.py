import torch
import torch.nn as nn
from torchvision.models import ResNet34_Weights, resnet34

from sdd.model.models.base.fpn import Fpn


class ResNetFPN(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.fpn_depth = 128

        resnet = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)

        self.adpater = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )  # /4 -> /4

        self.down1 = resnet.layer1  # /1 -> /4
        self.down2 = resnet.layer2  # /2 -> /8
        self.down3 = resnet.layer3  # /2 -> /16
        self.down4 = resnet.layer4  # /2 -> /32

        self.up1 = nn.Conv2d(512, self.fpn_depth, kernel_size=1)  # x1 -> /32
        self.up2 = Fpn(256, self.fpn_depth)  # x2 -> /16
        self.up3 = Fpn(128, self.fpn_depth)  # x2 -> /8
        self.up4 = Fpn(64, self.fpn_depth)  # x2 -> /4

        # self.fc = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(out_channels),
        #     nn.Flatten(),
        #     nn.Linear((out_channels**2) * self.fpn_depth, out_channels * res),
        # )

    def forward(self, x):  # (B, 3, H, W)
        p1 = self.adpater(x)  # (B, 64, H/4, W/4)

        p2 = self.down1(p1)  # (B, 64, H/4, W/4)
        p3 = self.down2(p2)  # (B, 128, H/8, W/8)
        p4 = self.down3(p3)  # (B, 256, H/16, W/16)
        p5 = self.down4(p4)  # (B, 512, H/32, W/32)

        f4 = self.up1(p5)  # (B, 128, H/32, W/32)
        f3 = self.up2(f4, p4)  # (B, 128, H/16, W/16)
        f2 = self.up3(f3, p3)  # (B, 128, H/8, W/8)
        f1 = self.up4(f2, p2)  # (B, 128, H/4, W/4)

        return f1
