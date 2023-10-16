import cv2
import datasets
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.utils as U
from attrs import define, field
from torch.utils.data import Dataset

from sdd.data.maps.basic import min_max_annotations, rescale
from sdd.data.utils.originals import keep_originals
from sdd.model.label import LabelMap


def to_rgb(items):
    items["image"] = [item.convert("RGB") for item in items["image"]]
    return items


@define
class StanfordDogsDataset(Dataset):
    resize = field(default=-1)
    num_classes = field(default=-1)

    def __attrs_post_init__(self):
        self.__keep_original = False
        self.__demo = False

        self.dataset = datasets.load_dataset("Alanox/stanford-dogs", split="full")

        df = self.dataset.to_pandas()["target"]
        unique_targets = df.unique()

        if self.num_classes != -1:
            self.num_classes_ = self.num_classes

            np.random.seed(self.num_classes_)

            rand_targets = np.random.choice(
                unique_targets, size=self.num_classes_
            ).tolist()

            b = df.isin(rand_targets)
            idx = df[b].index

            self.dataset = self.dataset.select(idx)
        else:
            self.num_classes_ = len(unique_targets)

        self.pytorch_dataset = self.dataset.with_format("torch")

        self.label_map = LabelMap(self.dataset["target"])

    def original(self, keep=False):
        self.__keep_original = keep
        return self

    def demo(self, demo=True):
        self.__demo = demo
        return self

    def __getitem__(self, idx):
        item = self.pytorch_dataset[idx]

        output = {"index": torch.tensor(idx)}

        to_keep = []
        if self.__keep_original:
            to_keep.append("image")

        # Map to keep track of original data and copy item data
        output = keep_originals(output, item, to_keep)
        output["annotations"] = min_max_annotations(
            output["annotations"], output["image"].shape[:2]
        )

        image = output["image"]
        if self.resize > 0:
            output["image"] = (
                TF.resize(
                    image.permute(2, 0, 1), (self.resize, self.resize), antialias=True
                )
                / 255.0
            )

        output["annotations_unscaled"] = rescale(
            output["annotations"], output["image"].shape[1:]
        )

        if self.__demo:
            for shape, mask_name, annots in zip(
                [output["image"].shape, image.permute(2, 0, 1).shape],
                ["mask", "original_mask"],
                [
                    output["annotations_unscaled"],
                    rescale(output["annotations"], image.shape[:2]),
                ],
            ):
                # Create mask
                mask = np.zeros(shape).transpose(1, 2, 0).copy()
                for annotation in annots:
                    x1, y1, x2, y2 = annotation.numpy()
                    cv2.rectangle(
                        mask,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255),
                        thickness=-1,
                    )

                mask = torch.from_numpy(mask)
                output[mask_name] = mask.permute(2, 0, 1) / 255.0

        # TODO
        # Add transforms for training

        return output

    def __len__(self):
        return len(self.pytorch_dataset)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.transforms as T

    # Resize
    size = 512

    # Dataset loading
    dataset = StanfordDogsDataset(size, 5).original(True).demo()

    print(dataset)

    # Picking random image
    rand = torch.randint(len(dataset), (1,)).squeeze().item()
    print(f"{rand=}")

    item = dataset[56]

    print(item.keys())

    item["original_image"] = item["original_image"].permute(2, 0, 1)
    torch_image = item["original_image"]
    annots = item["annotations_unscaled"]

    resize = T.Resize((size, size), antialias=True)

    def draw_bb(image, fill=False):
        return U.draw_bounding_boxes(
            image, annots, labels=[item["target"] for _ in annots], width=2, fill=fill
        )

    # Creating visual representations
    item["mask_over"] = draw_bb(resize(torch_image.clone()), fill=True)
    item["mask_over_border"] = draw_bb(resize(torch_image.clone()))

    # Creating visual representations
    item["original_mask_over"] = draw_bb(torch_image.clone(), fill=True)
    item["original_over_border"] = draw_bb(torch_image.clone())

    fig, axs = plt.subplots(2, 4, figsize=(20, 8))
    axs = axs.flatten()

    fig.suptitle(f"{item['name']}, {item['target']}")
    for ax, name in zip(
        axs,
        [
            "original_image",
            "original_mask",
            "original_mask_over",
            "original_over_border",
            "image",
            "mask",
            "mask_over",
            "mask_over_border",
        ],
    ):
        ax.grid(False)
        ax.axis("off")
        ax.set_title(name)
        ax.imshow(item[name].permute(1, 2, 0))

    plt.show()
