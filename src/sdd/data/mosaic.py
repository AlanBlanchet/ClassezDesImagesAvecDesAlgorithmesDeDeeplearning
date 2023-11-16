import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.utils as U
from attrs import define
from torch.utils.data import Dataset

from sdd.data.maps.basic import rescale
from sdd.data.standard import StandardStanfordDogsDataset


@define
class MosaicStanfordDogsDataset(Dataset):
    mosaic_size: int

    original: StandardStanfordDogsDataset
    train_idx: list[int]
    val_idx: list[int]

    def __attrs_post_init__(self):
        self._create_map()

    def _create_map(self):
        self.idx_map = []

        indexes = self.train_idx.copy()

        while len(indexes) > 0:
            # Original train length
            n = len(indexes)

            # Random size to pick elements
            size = np.random.randint(1, 8)

            # Chose randomly from train
            train_rands = np.random.choice(
                indexes, size=min(size, n), replace=False
            ).tolist()

            self.idx_map.append(train_rands)

            [indexes.remove(rand) for rand in train_rands]

        n = len(self.idx_map)
        # Add val images
        [self.idx_map.append([idx]) for idx in self.val_idx]
        # New idx
        self.train_idx_ = np.arange(n)
        self.val_idx_ = np.arange(n, n + len(self.val_idx))

    def add_coord_to_mosaic(self, coords: list[list[int, int, int, int]], size: int):
        # Get biggest coord and split it to make space for new image

        if len(coords) == 0:
            coords.append([0, 0, size, size])
            return

        # Chose a coord to split on
        i = np.argmax([area(c) for c in coords])
        coord = coords[i]
        w, h = wh(coord)

        c = w if w > h else h
        center_size = c // 2
        r = np.random.normal(0, 1)
        center_size += int(min(max(r, -1), 1) * (center_size // 4))

        if w > h:
            coord[2] = coord[0] + center_size
            coords.append([coord[2], coord[1], coord[2] + c - center_size, coord[3]])
        else:
            coord[3] = coord[1] + center_size
            coords.append([coord[0], coord[3], coord[2], coord[3] + c - center_size])

    def to_mosaic(self, items):
        """
        Combine multiple dataset images into 1
        """

        n = len(items)

        coords = []

        for _ in range(n):
            self.add_coord_to_mosaic(coords, self.mosaic_size)

        # Create the images from coords
        mosaic = torch.zeros((3, *((self.mosaic_size,) * 2)))
        annotations = []
        for item, coord in zip(items, coords):
            img = item["image"]
            annots = item["annotations"]
            xmin, ymin, xmax, ymax = coord
            w, h = wh(coord)
            [annotations.append(self._translate(coord, annot)) for annot in annots]
            img = TF.resize(img, (h, w), antialias=False)
            mosaic[..., ymin:ymax, xmin:xmax] = img

        return mosaic, torch.tensor(annotations)

    def _translate(
        self, coord: list[int, int, int, int], annotation: list[int, int, int, int]
    ):
        annotation = np.array(annotation)
        ratio_coord = np.array(coord) / self.mosaic_size
        xmin, ymin, *_ = ratio_coord

        w, h = wh(ratio_coord)

        ax = w * annotation[[0, 2]] + xmin
        ay = h * annotation[[1, 3]] + ymin

        return [ax[0], ay[0], ax[1], ay[1]]

    def __getitem__(self, index):
        item_idx = self.idx_map[index]

        original_items = [self.original[idx] for idx in item_idx]

        image, annotations = self.to_mosaic(original_items)
        annotations_unscaled = rescale(annotations, image.shape[1:])

        targets = []
        [targets.extend(item["target"]) for item in original_items]

        name = ",".join([item["name"] for item in original_items])

        return {
            "index": index,
            "image": image,
            "annotations": annotations,
            "target": targets,
            "annotations_unscaled": annotations_unscaled,
            "name": name,
        }

    def __len__(self):
        return len(self.idx_map)


def wh(coord: list[int, int, int, int]):
    return coord[2] - coord[0], coord[3] - coord[1]


def area(coord: list[int, int, int, int]):
    w, h = wh(coord)
    return w * h


def ratio(coord: list[int, int, int, int]):
    w, h = wh(coord)
    return w / h


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Resize
    size = 1024

    # Dataset loading
    dataset = StandardStanfordDogsDataset(size).original(True).demo()

    np.random.seed(2)

    print(dataset)

    idx = list(np.arange(2000))

    dataset = MosaicStanfordDogsDataset(
        1024, original=dataset, train_idx=idx[:1500], val_idx=idx[1500:]
    )

    print(dataset)

    # Picking random image
    rand = torch.randint(len(dataset), (1,)).squeeze().item()
    print(f"{rand=}")

    train_map = dataset.idx_map[4]

    items = [dataset.original[idx] for idx in train_map]

    item = dataset[1]
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
