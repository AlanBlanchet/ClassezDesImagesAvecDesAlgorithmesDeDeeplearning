import torch
import torch.nn as nn

from src.model.models.test import Test


class StanfordDogsModel(torch.nn.Module):
    def __init__(self, img_size, num_classes, model_name="test"):
        super(StanfordDogsModel, self).__init__()

        self.img_size = img_size
        self.num_classes = num_classes
        self.model_name = model_name

        if self.model_name == "test":
            self.model = Test(self.img_size, self.num_classes)
        else:
            raise NotImplementedError

    def forward(self, batch: torch.Tensor):
        return self.model(batch)

    def __call__(self, batch):
        self.forward(batch)
