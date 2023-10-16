import torch

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
