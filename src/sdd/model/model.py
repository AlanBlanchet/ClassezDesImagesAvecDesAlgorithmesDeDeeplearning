import timm
import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange

from sdd.compat.utils import is_task_bb
from sdd.model.heads.head import ObjectClassificationHead, ObjectDetectionHead
from sdd.model.models.resnetfpn import ResNetFPN
from sdd.model.models.test import Test
from sdd.model.models.vgg import VGG


class StanfordDogsModel(torch.nn.Module):
    def __init__(
        self, config, img_size, num_classes, num_bb, model_name="custom::test"
    ):
        super(StanfordDogsModel, self).__init__()

        self.img_size = img_size
        self.num_classes = num_classes
        self.model_name = model_name
        self.num_bb = num_bb
        # self.head_in = 16
        # self.wh = 8
        # self.res = self.wh**2

        task = config.get("task", "detection")
        is_bb = is_task_bb(task)

        self.in_channels = None

        self.model_prefix = ""
        model = None

        model_name = self.model_name
        source, model_name = model_name.split("::")

        if source == "custom":
            if model_name == "test":
                model = Test(self.num_classes, self.num_bb)
                self.in_channels = 96
            elif model_name == "resnetfpn":
                model = ResNetFPN()
                self.in_channels = 128
            elif model_name.startswith("vgg"):
                postfix = model_name.split(":")[1]
                if is_bb:
                    raise ValueError("VGG isn't made for object detection")
                model = nn.Sequential(VGG(postfix), nn.Linear(4096, num_classes))
            else:
                raise NotImplementedError
        elif source == "torchvision":
            if model_name == "alexnet":
                model = (
                    nn.Sequential(
                        models.AlexNet().features, nn.AdaptiveAvgPool2d((6, 6))
                    )
                    if is_bb
                    else models.AlexNet(num_classes=num_classes)
                )
                if is_bb:
                    self.in_channels = 256
            elif model_name == "vgg16":
                model = models.vgg16().features
                self.in_channels = 512
            elif model_name == "resnet":
                self.in_channels = 64
                model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
                model.fc = nn.Linear(model.fc.in_features, self.in_channels * self.res)
            else:
                raise NotImplementedError
        elif source == "timm":
            model = timm.create_model(
                model_name,
                num_classes=self.in_channels * self.res,
                img_size=self.img_size,
            )
            self.head_type = "l"
            self.in_channels = 32
        else:
            raise NotImplementedError

        self.head = None
        self.is_bb = is_bb

        if self.in_channels is not None:
            if is_bb:
                self.head = ObjectDetectionHead(self.in_channels, num_bb, num_classes)
            else:
                self.head = ObjectClassificationHead(self.in_channels, num_classes)

        self.model = model

    def forward(self, x: torch.Tensor):
        B, *_ = x.shape  # B C H W

        x = self.model(x)  # B ----

        if self.head is None:
            return x

        return self.head(x)
        # if self.model_prefix != "":
        #     x = x.reshape((-1, self.features_in, self.wh, self.wh))
        # x = self.head(x)
        # return x.view(B, self.num_bb, self.num_classes)
