import cv2
import datasets
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.utils as U
from attrs import define, field
from torch.utils.data import Dataset

from sdd.data.augmentations import StanfordDogsAugmentations
from sdd.data.maps.basic import min_max_annotations, rescale
from sdd.data.utils.originals import keep_originals
from sdd.model.label import LabelMap


def to_rgb(items):
    items["image"] = [item.convert("RGB") for item in items["image"]]
    return items


@define
class StandardStanfordDogsDataset(Dataset):
    img_size = field(default=-1)
    num_classes = field(default=-1)
    augmentations = field(default=None)
    base_augmentations = field(default=None)

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
                unique_targets, size=self.num_classes_, replace=False
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

    def set_base_augmentations(self, augmentations):
        self.base_augmentations = augmentations

    def __getitem__(self, idx):
        item = self.pytorch_dataset[int(idx)]

        output = {"index": torch.tensor(idx)}

        to_keep = []
        if self.__keep_original:
            to_keep.extend(["image", "annotations"])

        # Map to keep track of original data and copy item data
        output = keep_originals(output, item, to_keep)

        if self.__keep_original:
            original_image = output["original_image"]
        image = output["image"].permute(2, 0, 1)

        output["annotations"] = min_max_annotations(
            output["annotations"], image.shape[1:]
        ).clamp_max(1.0)
        output["target"] = [item["target"] for _ in range(len(output["annotations"]))]

        if self.img_size > 0:
            image = TF.resize(
                image,
                (self.img_size, self.img_size),
                antialias=True,
            )

        if self.augmentations:
            result = self.augmentations(
                image=image.permute(1, 2, 0),
                bbox=output["annotations"],
                labels=output["target"],
            )
            image = torch.from_numpy(result["image"]).permute(2, 0, 1) / 255.0
            output["annotations"] = torch.tensor(result["bboxes"])

        if self.base_augmentations:
            image = self.base_augmentations(image)

        output["annotations_unscaled"] = rescale(output["annotations"], image.shape[1:])

        if self.__demo and self.__keep_original:
            for shape, mask_name, annots in zip(
                [image.permute(1, 2, 0).shape, original_image.shape],
                ["mask", "original_mask"],
                [output["annotations_unscaled"], output["original_annotations"]],
            ):
                # Create mask
                mask = np.zeros(shape).copy()
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

        # image = self.normalize(image)

        output["image"] = image
        return output

    def __len__(self):
        return len(self.pytorch_dataset)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.transforms as T

    # Resize
    size = 512

    # Augs
    augs = StanfordDogsAugmentations()

    # Dataset loading
    dataset = StandardStanfordDogsDataset(size, 5, augs).original(True).demo()

    print(dataset)

    # Picking random image
    rand = torch.randint(len(dataset), (1,)).squeeze().item()
    print(f"{rand=}")

    item = dataset[56]

    print(item.keys())

    item["original_image"] = item["original_image"].permute(2, 0, 1)
    original_image = item["original_image"]
    original_scaled_annots = item["original_annotations"]
    image = (item["image"] * 255).to(torch.uint8)
    annots = item["annotations"]
    scaled_annots = rescale(annots, image.shape[1:])

    resize = T.Resize((size, size), antialias=True)

    def draw_bb(image, annotations, fill=False):
        return U.draw_bounding_boxes(
            image, annotations, labels=item["target"], width=2, fill=fill
        )

    # Creating visual representations
    item["mask_over"] = draw_bb(image.clone(), scaled_annots, fill=True)
    item["mask_over_border"] = draw_bb(image.clone(), scaled_annots)

    # Creating visual representations
    item["original_mask_over"] = draw_bb(
        original_image.clone(), original_scaled_annots, fill=True
    )
    item["original_over_border"] = draw_bb(
        original_image.clone(), original_scaled_annots
    )

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

    # Augmentations
    rows, cols = 4, 7
    fig, axs = plt.subplots(rows, cols, figsize=(20, 8))
    axs = axs.flatten()

    fig.suptitle(f"{item['name']}, {item['target']}")
    for i, ax in enumerate(axs, start=1):
        item = dataset[56]
        image = (item["image"] * 255).to(torch.uint8)
        ax.grid(False)
        ax.axis("off")
        ax.set_title(f"Aug {i}")
        ax.imshow(image.permute(1, 2, 0))

    plt.show()
