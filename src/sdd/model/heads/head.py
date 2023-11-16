import torch
import torch.nn as nn

from .box import BoxHead
from .classification import ClassificationHead
from .objectness import ObjectnessHead


class ObjectDetectionHead(nn.Module):
    def __init__(self, in_chans: int, num_bb: int, num_classes: int):
        super().__init__()

        self.classifier = ClassificationHead(in_chans, num_classes, num_bb)

        self.objectness = ObjectnessHead(in_chans, num_bb)

        self.box = BoxHead(in_chans, num_bb)

    def forward(self, x):
        out_clf = self.classifier(x)  # (B, N, S, C)
        out_clf = out_clf.mean(dim=-2)  # (B, N, C)

        out_objn = self.objectness(x)  # (B, N, S, 1)
        out_objn = out_objn.mean(dim=-2).squeeze(dim=-1)  # (B, N)

        out_bb = self.box(x)  # (B, N, S, 4)
        out_bb = out_bb.mean(dim=-2)  # (B, N, 4)

        return out_clf, out_bb, out_objn


class ObjectClassificationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        self.classifier = ClassificationHead(in_channels, num_classes)

    def forward(self, x):
        shape = x.shape

        if len(shape) == 4:
            out_clf: torch.Tensor = self.classifier(x)  # (B, N, S, C)
        else:
            out_clf = self.classifier(x.view(shape[0], 1, 1, shape[1]))

        out_clf = out_clf.mean(dim=(-3, 2))  # (B, C)

        return out_clf
