import torch.nn as nn
import torchvision.models as models


class FasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = models.detection.fasterrcnn_resnet50_fpn(num_classes=num_classes)

        # self.transform = self.model.transform

        # images, targets = self.transform(images, targets)

    def forward(self, x, targets):
        # Custom
        output = self.model(x, targets)

        t = type(output)
        print("=" * 20)
        print(t.__name__)

        if isinstance(output, list):
            print(type(output[0]).__name__)

        clf_loss = output["loss_classifier"]
        rpn_bb_loss = output["loss_rpn_box_reg"]
        bb_loss = output["loss_box_reg"]
        obj_loss = output["loss_objectness"]

        bb_loss += rpn_bb_loss

        loss_sum = clf_loss + bb_loss + obj_loss

        return loss_sum, (clf_loss, bb_loss, obj_loss)
