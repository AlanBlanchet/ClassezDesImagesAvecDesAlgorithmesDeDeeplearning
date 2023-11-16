import torch
import torch.nn as nn
import torch.nn.functional as F

from sdd.model.heads import ObjectDetectionHead


class Test(nn.Module):
    def __init__(self, num_classes: int, num_bb: int):
        super(Test, self).__init__()

        size = 224

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 48, 3, padding=1)
        self.conv3 = nn.Conv2d(48, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 80, 3, padding=1)
        self.conv5 = nn.Conv2d(80, 96, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten(start_dim=1)

        hidden = (size // (2**5)) ** 2 * 96

        self.fc = nn.Linear(hidden, num_bb * num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))  # B Cls 7 7
        return x
        # x = self.flatten(x)
        # return self.fc(x)
