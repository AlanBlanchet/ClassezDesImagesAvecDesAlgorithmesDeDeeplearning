from copy import deepcopy

import timm
import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange
from timm.models.resnet import default_cfgs
from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
)

from sdd.compat.utils import is_task_bb
from sdd.model.heads.head import ObjectClassificationHead, ObjectDetectionHead
from sdd.model.models import HF, VGG, ResNetFPN, Test, TimmModel
from sdd.model.models.wrappers.torchvision.faster_rcnn import FasterRCNN


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
        pretrained = config.get("pretrained", False)

        self.format_kwargs = None

        self.in_channels = None

        model = None

        model_name = self.model_name
        model_sup = None
        source, *model_name = model_name.split("::")

        if len(model_name) > 1:
            model_sup, model_name = model_name
        else:
            model_name = model_name[0]

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
            if model_sup == "detection":
                if model_name == "faster-rcnn":
                    model = FasterRCNN(num_classes=num_classes)

                    def _format_kwargs(bbs, clfs, objs):
                        items = []
                        for bb, clf, obj in zip(bbs, clfs, objs):
                            mask = obj == 1
                            bb = bb[mask]
                            clf = clf[mask]

                            items.append({"labels": clf.to(torch.int64), "boxes": bb})
                        return {"targets": items}

                    self.format_kwargs = _format_kwargs
                else:
                    raise NotImplementedError
            elif model_name == "alexnet":
                weights = models.AlexNet_Weights.DEFAULT if pretrained else None
                model = (
                    nn.Sequential(
                        models.alexnet(weights).features, nn.AdaptiveAvgPool2d((6, 6))
                    )
                    if is_bb
                    else models.alexnet(weights, num_classes=num_classes)
                )
                if is_bb:
                    self.in_channels = 256
            elif model_name == "vgg16":
                weights = models.VGG16_Weights.DEFAULT if pretrained else None
                model = (
                    nn.Sequential(
                        models.vgg16(weights).features, nn.AdaptiveAvgPool2d((6, 6))
                    )
                    if is_bb
                    else models.vgg16(weights, num_classes=num_classes)
                )
                if is_bb:
                    self.in_channels = 512
            elif model_name == "resnet":
                weights = models.ResNet34_Weights.DEFAULT if pretrained else None
                model = models.resnet34(weights)
                model = nn.Sequential(
                    model.conv1,
                    model.bn1,
                    model.relu,
                    model.maxpool,
                    model.layer1,
                    model.layer2,
                    model.layer3,
                    model.layer4,
                )
                self.in_channels = 512
            else:
                raise NotImplementedError
        elif source == "timm":
            model = timm.create_model(
                model_name,
                num_classes=num_classes,
                pretrained=pretrained,
                features_only=is_bb,
            )
            if is_bb:
                self.in_channels = model.feature_info.channels()[-1]
                model = TimmModel(model)
        elif source == "transformers":
            if not pretrained:
                print(
                    "Hugging Face models aren't to be used un-pretrained since it is hard to manage"
                )
                print("Using a pretrained model by default")

            if is_bb:
                model = AutoModelForObjectDetection.from_pretrained(
                    model_name,
                    num_labels=self.num_classes,
                    ignore_mismatched_sizes=True,
                    num_queries=8,
                )
            else:
                model = AutoModelForImageClassification.from_pretrained(
                    model_name,
                    num_labels=self.num_classes,
                    ignore_mismatched_sizes=True,
                )
            model = HF(model, task)
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

    def forward(self, x: torch.Tensor, **kwargs):
        B, *_ = x.shape  # B C H W

        kwargs = self.format_kwargs(**kwargs) if self.format_kwargs is not None else {}

        x = self.model(x, **kwargs)  # B ----

        if self.head is None:
            return x

        return self.head(x, **kwargs)
