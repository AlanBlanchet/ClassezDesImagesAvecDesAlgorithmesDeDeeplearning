import torch
import torch.nn as nn

from sdd.model.models.tools import SelectiveSearch


class RCNN:
    def __init__(self, num_classes: int, num_bb: int, extractor="vgg"):
        super().__init__()

        self.search = SelectiveSearch()

        self.conv1 = nn.Conv2d(
            256,
        )

        self.fc1 = nn.Linear(0, 4096)
        self.fc2 = nn.Linear(4096, 4096)

    def forward(self, x):
        # (B, 3, H, W)

        _, regions, _ = self.search(img_rgbb3hw_1=x)

        for region in regions:
            annots = torch.tensor([r["bbox_xywh"] for r in region])

        x -= x.mean()


if __name__ == "__main__":
    import datasets
    import numpy as np
    from torchvision.transforms.functional import resize, to_pil_image
    from torchvision.utils import draw_bounding_boxes

    from sdd.compat import to_format

    data = datasets.load_dataset("mnist")

    img = data["train"][0]["image"]

    size = (256, 256)

    img = np.array(img)
    img = np.stack([img, img, img], axis=0)
    img = resize(torch.from_numpy(img), size, antialias=True)
    img = img.unsqueeze(dim=0) / 255

    print(img.shape)

    search = SelectiveSearch()

    boxes_xywh, regions, reg_lab = search(img_rgbb3hw_1=img)

    [print(b.shape) for b in boxes_xywh]

    # bbox_xywh, region_size, rank
    annots = torch.tensor([r["bbox_xywh"] for r in regions[0]])
    print(annots.shape)
    annots = to_format(annots, size, "pascal_voc")

    img = img * 255
    img = img.to(torch.uint8)
    img = draw_bounding_boxes(img.squeeze(dim=0), annots, colors="red")
    to_pil_image(img).show()
