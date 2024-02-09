from typing import Literal

import albumentations as A
import torch
import torchvision.transforms.functional as TF

FORMATS = Literal["pascal_voc", "coco", "yolo"]

default_resize_order = [1, 0]


def min_max_annotations(
    annotations: torch.Tensor, size: list[int, int], order=default_resize_order
):
    size = torch.tensor(size, device=annotations.device)
    annotations = annotations / size[order * 2]
    return annotations


def rescale(
    annotations: torch.Tensor, size: list[int, int], order=default_resize_order
):
    size = torch.tensor(size, device=annotations.device)
    annotations = annotations * size[order * 2]
    return annotations.round().int()


def to_format(
    annotations,
    size: list[int, int],
    wanted_format: FORMATS,
    current_format: FORMATS = "pascal_voc",
    validate=True,
):
    if validate:
        annotations = validate_annotations(annotations, current_format, wh=size)

    if current_format == wanted_format:
        return annotations

    if current_format == "pascal_voc":
        annots = annotations.T

        xmin, ymin, xmax, ymax = annots[0:1], annots[1:2], annots[2:3], annots[3:4]

        w, h = (xmax - xmin, ymax - ymin)

        if wanted_format == "coco":
            annotations = torch.cat([xmin.T, ymin.T, w.T, h.T], dim=1)
        elif wanted_format == "yolo":
            cx, cy = xmin + w / 2, ymin + h / 2
            annotations = min_max_annotations(
                torch.cat([cx.T, cy.T, w.T, h.T], dim=1), size=size
            )
        else:
            raise ValueError
    elif current_format == "coco":
        if wanted_format == "pascal_voc":
            annotations[..., 2:] = annotations[..., :2] + annotations[..., 2:]
        elif wanted_format == "yolo":
            annotations = to_format(annotations, size, "pascal_voc", current_format)
            annotations = to_format(annotations, size, wanted_format, "pascal_voc")
        else:
            raise ValueError
    elif current_format == "yolo":
        if wanted_format == "pascal_voc":
            half = annotations[..., 2:] / 2
            annotations[..., 2:] = annotations[..., :2] + half
            annotations[..., :2] = annotations[..., :2] - half
            annotations = rescale(annotations, size=size)
        elif wanted_format == "coco":
            annotations = to_format(annotations, size, "pascal_voc", current_format)
            annotations = to_format(annotations, size, wanted_format, "pascal_voc")
        else:
            raise ValueError
    else:
        raise ValueError

    print("VALIDATE", validate, size)
    if validate:
        annotations = validate_annotations(annotations, wanted_format, wh=size)

    return annotations


def to_normalized(annotations, format: FORMATS, size):
    if format != "yolo":
        annotations = min_max_annotations(annotations, (size, size))
    return annotations


def to_unnormalized(annotations, format: FORMATS, size):
    if format != "yolo":
        annotations = rescale(annotations, size)
    return annotations


mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
unnormalize = A.Normalize(mean=-mean / std, std=1 / std, max_pixel_value=1)


def format_for_visualize(image, annotations, format, size, normalized=False):
    image = image.clone()
    annotations = annotations.clone()
    shape = image.shape

    if shape[0] == 3:
        image = image.permute(1, 2, 0)

    annotations = validate_annotations(
        annotations, format, normalized=normalized, wh=(1, 1)
    )

    image = image.cpu().numpy()

    img_size = image.shape[:2]

    if normalized:
        image = unnormalize(image=image)["image"]
        image *= 255
        annotations = to_unnormalized(annotations, format, img_size)

    annotations = to_format(annotations, img_size, "pascal_voc", format)
    annotations = annotations.cpu()

    resize = A.Compose(
        [A.Resize(width=size, height=size)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=[]),
    )

    result = resize(image=image, bboxes=annotations.numpy())

    annotations = torch.tensor(result["bboxes"])
    image = torch.from_numpy(result["image"]).to(torch.uint8)

    if shape[0] == 3:
        image = image.permute(2, 0, 1)

    return image, annotations


def validate_annotations(annotations, format, normalized=False, wh=None):
    annotations = torch.clip(annotations, min=0)

    if wh is not None:
        h, w = wh
        annotations[..., [0, 2]] = torch.clip(annotations[..., [0, 2]], min=0, max=w)
        annotations[..., [1, 3]] = torch.clip(annotations[..., [1, 3]], min=0, max=h)

    if format == "pascal_voc":
        # Validate annotations
        for bb in annotations:
            step = normalized_step(normalized)
            if bb[0] > bb[2]:
                bb[[0, 2]] = bb[[2, 0]]
            elif bb[0] == bb[2]:
                idx = 0 if bb[2] == wh[0] else 2
                bb[idx] += -step if idx == 0 else step
            if bb[1] > bb[3]:
                bb[[1, 3]] = bb[[3, 1]]
            elif bb[1] == bb[3]:
                idx = 1 if bb[3] == wh[1] else 3
                bb[idx] += -step if idx == 1 else step
    elif format == "coco":
        pascal = to_format(annotations, wh, "pascal_voc", format, validate=False)
        annotations = validate_annotations(pascal, "pascal_voc", normalized=normalized)
        annotations = to_format(annotations, wh, format, "pascal_voc")
    elif format == "yolo":
        if wh is None:
            raise ValueError("Yolo format requires original size")
        pascal = to_format(annotations, wh, "pascal_voc", format, validate=False)
        annotations = validate_annotations(pascal, "pascal_voc", normalized=normalized)
        annotations = to_format(annotations, wh, format, "pascal_voc")

    if normalized:
        annotations = torch.clip(annotations, min=0, max=1)

    return annotations


def normalized_step(normalized=True):
    return 0.01 if normalized else 1
