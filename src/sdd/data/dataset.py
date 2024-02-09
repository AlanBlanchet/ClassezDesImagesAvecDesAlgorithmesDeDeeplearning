from attrs import define, field
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from sdd.data.augmentations import BaseAugmentations, StanfordDogsAugmentations
from sdd.data.mosaic import MosaicStanfordDogsDataset
from sdd.data.standard import StandardStanfordDogsDataset


@define
class StanfordDogsDataset(Dataset):
    config = field()
    img_size = field(default=-1)
    num_classes = field(default=-1)
    augmentations = field(default=None)
    mosaic = field(default=False)

    def __attrs_post_init__(self):
        self._std_dataset = StandardStanfordDogsDataset(
            self.img_size,
            num_classes=self.num_classes,
            augmentations=self.augmentations,
            format=self.config.get("format", "pascal_voc"),
        )
        self.num_classes_ = self._std_dataset.num_classes_
        self.label_map = self._std_dataset.label_map

        idx_train, idx_test = train_test_split(
            range(len(self._std_dataset)),
            stratify=self._std_dataset.dataset["target"],
            random_state=0,
        )

        # Base augmentations
        base_augmentations = BaseAugmentations(
            self.config, self.img_size, self._std_dataset.pytorch_dataset, idx_train
        )
        self._std_dataset.set_base_augmentations(base_augmentations)

        if self.mosaic:
            self.dataset_ = MosaicStanfordDogsDataset(
                self.img_size, self._std_dataset, idx_train, idx_test
            )
            self.train_idx_ = self.dataset_.train_idx_
            self.val_idx_ = self.dataset_.val_idx_
        else:
            self.dataset_ = self._std_dataset
            self.train_idx_ = idx_train
            self.val_idx_ = idx_test

    def original(self, keep: bool = False):
        self._std_dataset.original(keep)
        return self

    def demo(self, demo: bool = True):
        self._std_dataset.demo(demo)
        return self

    def train(self, train=True):
        self.dataset_.train(train)

    def set_base_augmentations(self, augmentations):
        self.dataset_.set_base_augmentations(augmentations)

    def __len__(self):
        return len(self.dataset_)

    def __getitem__(self, idx):
        return self.dataset_[idx]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    import torchvision.transforms.functional as TF
    import torchvision.utils as U

    # Resize
    size = 1024
    n_cls = 5

    # Augs
    augs = StanfordDogsAugmentations()

    # Dataset loading
    dataset = StanfordDogsDataset(size, n_cls, augs).original(True).demo()

    print(dataset)

    item = dataset[-1]
    image, annots = item["image"], item["annotations_unscaled"]
    image = (image * 255).to(torch.uint8)
    img = TF.to_pil_image(image)

    def draw_bb(image, fill=False):
        return U.draw_bounding_boxes(
            image, annots, labels=[item["target"] for _ in annots], width=2, fill=fill
        )

    # Creating visual representations
    item["mask_over"] = draw_bb(image.clone(), fill=True)
    item["mask_over_border"] = draw_bb(image.clone())

    fig, axs = plt.subplots(1, 3, figsize=(20, 8))
    axs = axs.flatten()

    print("SHAPE", image.dtype, image.min(), image.max())

    fig.suptitle(f"{item['name']}, {','.join(item['target'])}")
    for ax, name in zip(
        axs,
        [
            "image",
            "mask_over",
            "mask_over_border",
        ],
    ):
        ax.grid(False)
        ax.axis("off")
        ax.set_title(name)
        ax.imshow(item[name].permute(1, 2, 0))

    plt.show()
