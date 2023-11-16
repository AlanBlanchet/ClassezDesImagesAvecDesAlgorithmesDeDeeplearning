import torch.nn as nn
from einops import rearrange


class ClassificationHead(nn.Module):
    def __init__(self, in_chans: int, num_classes: int, num_bb: int = 1):
        super().__init__()

        self.num_classes = num_classes
        num_chans = num_bb * num_classes

        self.dropout = nn.Dropout()
        self.conv = nn.Conv2d(in_chans, num_chans, 1)

    def forward(self, x):
        x = self.conv(self.dropout(x))  # (B, C, H, W)
        return rearrange(x, "b (num_bb p) h w -> b num_bb (h w) p", p=self.num_classes)
