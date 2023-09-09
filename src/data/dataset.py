from collections import defaultdict
from typing import Literal

import datasets
from attrs import define, field
from torch.utils.data import Dataset
from tqdm import tqdm

from src.model.label import LabelMap


def to_rgb(items):
    items["image"] = [item.convert("RGB") for item in items["image"]]
    return items


@define
class StanfordDogsDataset(Dataset):
    base_transforms = field()
    dataset = datasets.load_dataset("Alanox/stanford-dogs", split="full")
    pytorch_dataset = dataset.with_format("torch")
    label_map = LabelMap(dataset["target"])
    mode: Literal["train", "val"] = "train"

    def eval(self):
        self.mode = "val"

    def train(self):
        self.mode = "train"

    def __getitem__(self, idx):
        item = self.pytorch_dataset[idx]

        output = {}
        image = item["image"].permute(2, 0, 1)
        output["image"] = self.base_transforms(image) / 255.0

        output["target"] = item["target"]

        # TODO
        # Add transforms for training

        return output

    def __len__(self):
        return len(self.pytorch_dataset)


if __name__ == "__main__":
    dataset = StanfordDogsDataset()
    print(dataset)
