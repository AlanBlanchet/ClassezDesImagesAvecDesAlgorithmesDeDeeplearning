import torch
import torch.nn as nn
import torch.nn.functional as F

from sdd.model.heads import ObjectDetectionHead


class Test(nn.Module):
    def __init__(self, s_in: int, num_classes: int, num_bb: int):
        super(Test, self).__init__()

        self.s_in = s_in
        self.num_bb = num_bb

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x
