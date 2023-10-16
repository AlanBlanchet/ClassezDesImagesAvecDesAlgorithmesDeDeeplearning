import torch.nn as nn
from einops import rearrange


class ObjectnessHead(nn.Module):
    def __init__(self, in_chans: int, num_bb: int):
        super().__init__()

        self.conv = nn.Conv2d(in_chans, num_bb, 1)

    def forward(self, x):
        x = self.conv(x)
        return rearrange(x, "b (num_bb p) h w -> b num_bb (h w) p", p=1)
