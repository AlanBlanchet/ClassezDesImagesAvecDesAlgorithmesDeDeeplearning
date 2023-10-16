import torch.nn as nn

from . import ClassificationHead, ObjectnessHead, RegressionHead


class ObjectDetectionHead(nn.Module):
    def __init__(self, in_chans: int, num_bb: int, num_classes: int):
        self.classifier = ClassificationHead(in_chans, num_bb, num_classes)

        self.objectness = ObjectnessHead(in_chans, num_bb)

        self.regression = RegressionHead()

    def forward(self, x):
        out_clf = self.classifier(x)  # (B, N, S, C)

        out_objn = self.objectness(x)  # (B, N, S, 1)

        out_reg = self.regression(x)  # (B, N, S, 4)

        return out_clf, out_objn, out_reg
