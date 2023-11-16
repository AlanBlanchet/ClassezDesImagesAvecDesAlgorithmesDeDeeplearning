"""
My implementation of the VGG Network from the original paper
https://arxiv.org/pdf/1409.1556.pdf?ref=blog.paperspace.com
"""
import torch.nn as nn

configs = {
    "A": [
        ["C", 3, 64],
        "M",
        ["C", 64, 128],
        "M",
        ["C", 128, 256],
        ["C", 256, 256],
        "M",
        ["C", 256, 512],
        ["C", 512, 512],
        "M",
        ["C", 512, 512],
        ["C", 512, 512],
        "M",
    ],
    "A-LRN": [
        ["C", 3, 64],
        "L",
        "M",
        ["C", 64, 128],
        "M",
        ["C", 128, 256],
        ["C", 256, 256],
        "M",
        ["C", 256, 512],
        ["C", 512, 512],
        "M",
        ["C", 512, 512],
        ["C", 512, 512],
        "M",
    ],
    "B": [
        ["C", 3, 64],
        ["C", 64, 64],
        "L",
        "M",
        ["C", 64, 128],
        ["C", 128, 128],
        "M",
        ["C", 128, 256],
        ["C", 256, 256],
        "M",
        ["C", 256, 512],
        ["C", 512, 512],
        "M",
        ["C", 512, 512],
        ["C", 512, 512],
        "M",
    ],
    "C": [
        ["C", 3, 64],
        ["C", 64, 64],
        "L",
        "M",
        ["C", 64, 128],
        ["C", 128, 128],
        "M",
        ["C", 128, 256],
        ["C", 256, 256],
        ["C", 256, 256, 1],
        "M",
        ["C", 256, 512],
        ["C", 512, 512],
        ["C", 512, 512, 1],
        "M",
        ["C", 512, 512],
        ["C", 512, 512],
        ["C", 512, 512, 1],
        "M",
    ],
    "D": [
        ["C", 3, 64],
        ["C", 64, 64],
        "L",
        "M",
        ["C", 64, 128],
        ["C", 128, 128],
        "M",
        ["C", 128, 256],
        ["C", 256, 256],
        ["C", 256, 256],
        "M",
        ["C", 256, 512],
        ["C", 512, 512],
        ["C", 512, 512],
        "M",
        ["C", 512, 512],
        ["C", 512, 512],
        ["C", 512, 512],
        "M",
    ],
    "E": [
        ["C", 3, 64],
        ["C", 64, 64],
        "L",
        "M",
        ["C", 64, 128],
        ["C", 128, 128],
        "M",
        ["C", 128, 256],
        ["C", 256, 256],
        ["C", 256, 256],
        ["C", 256, 256],
        "M",
        ["C", 256, 512],
        ["C", 512, 512],
        ["C", 512, 512],
        ["C", 512, 512],
        "M",
        ["C", 512, 512],
        ["C", 512, 512],
        ["C", 512, 512],
        ["C", 512, 512],
        "M",
    ],
}


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=1 if kernel_size > 1 else 0,
        )
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class VGG(nn.Module):
    def __init__(self, type="A", in_channels=3, size=224):
        super().__init__()

        self.features = self._generate(type)

        self.fc1 = nn.Linear(512 * (size // (2**5)) ** 2, 4096)
        self.fc2 = nn.Linear(4096, 4096)

    def _generate(self, type: str):
        arch = configs[type]

        layers = []
        for item in arch:
            if isinstance(item, str):
                if item == "M":
                    layers.append(nn.MaxPool2d(2))
                elif item == "L":
                    layers.append(nn.LocalResponseNorm(2))
            elif isinstance(item, list):
                t, *ps = item
                if t == "C":
                    layers.append(ConvBlock(*ps))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.pool(self.conv1(x))  # WH = 112

        # x = self.pool(self.conv2(x))  # WH = 56

        # x = self.pool(self.conv3_2(self.conv3_1(x)))  # WH = 28

        # x = self.pool(self.conv4_2(self.conv4_1(x)))  # WH = 14

        # x = self.pool(self.conv5_2(self.conv5_2(x)))  # WH = 7
        x = self.features(x)

        x = x.flatten(start_dim=1)

        return self.fc2(self.fc1(x))
