import timm
import torch
import torchvision.models as models

from sdd.model.heads.head import ObjectDetectionHead
from sdd.model.models.test import Test


class StanfordDogsModel(torch.nn.Module):
    def __init__(self, img_size, num_classes, num_bb, model_name="test"):
        super(StanfordDogsModel, self).__init__()

        self.img_size = img_size
        self.num_classes = num_classes
        self.model_name = model_name
        self.num_bb = num_bb
        self.head_in = 64

        self.model_prefix = ""
        if self.model_name == "test":
            self.model = Test(self.img_size, self.num_classes, self.num_bb)
        elif self.model_name.lower() == "alexnet":
            self.model = models.AlexNet().features
            self.head_in = 256
        elif self.model_name.lower() == "vgg16":
            self.model = models.vgg16().features
            self.head_in = 512
        elif self.model_name.lower().startswith("timm:"):
            self.model_prefix = self.model_name.split("timm:")[1].lower()
            self.head_in = 32
            self.model = timm.create_model(
                self.model_prefix,
                num_classes=self.head_in * (32**2),
                img_size=self.img_size,
            )
            self.head_type = "l"
        else:
            raise NotImplementedError

        self.head = ObjectDetectionHead(self.head_in, num_bb, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        if self.model_prefix != "":
            x = x.reshape((-1, self.head_in, 32, 32))
        x = self.head(x)
        return x
