import torch
import torch.nn as nn
import torch.nn.functional as F

from sdd.compat.utils import chose_if_task_bb, is_task_bb

from .losses import *


class ObjectDetectionLoss(nn.Module):
    def __init__(self, img_size, format):
        super().__init__()

        self.img_size = img_size
        self.format = format

    def forward(self, clf, box, obj):
        # (B, N, ?)

        out_clf, true_clf = clf
        out_bbs, true_bbs = box
        out_obj, true_obj = obj

        mask = true_obj == 1

        clf_loss = F.cross_entropy(out_clf[mask], true_clf[mask])

        bb_loss = F.l1_loss(out_bbs[mask], true_bbs[mask]) * 4

        obj_loss = F.mse_loss(out_obj.sigmoid(), true_obj) * 2

        loss = clf_loss + bb_loss + obj_loss

        return loss, (clf_loss, bb_loss, obj_loss)


class ObjectClassificationLoss(nn.Module):
    def __init__(self, img_size, format=None):
        super().__init__()

        self.img_size = img_size

    def forward(self, clf):
        out_clf, true_clf = clf

        return F.cross_entropy(out_clf, true_clf)


def get_loss(loss_name, task, img_size, format):
    if loss_name is None:
        return chose_if_task_bb(ObjectDetectionLoss, ObjectClassificationLoss, task)(
            img_size, format
        )
    elif loss_name == "faster-rcnn":
        if is_task_bb(task):
            return FastRCNNLoss()
        raise ValueError("No classification for FasterRCNN")
    else:
        raise NotImplementedError(
            f"{loss_name} doesn't correspond to any implemented loss"
        )


if __name__ == "__main__":
    n_classes = 2
    n_boxes = 3

    loss = ObjectDetectionLoss()

    out_clf = torch.randint(n_classes, (2, n_boxes, n_classes)).float()
    true_clf = torch.randint(n_classes, (2, n_boxes))

    print(f"{out_clf.shape=} {true_clf.shape=}")

    out_obj = torch.randint(2, (2, n_boxes)).float()
    true_obj = torch.randint(2, (2, n_boxes))
    print(f"{out_obj.shape=} {true_obj.shape=}")

    out_bb = torch.randn((2, n_boxes, 4)).float()
    true_bb = torch.randn((2, n_boxes, 4))
    print(f"{out_bb.shape=} {true_bb.shape=}")

    print("\nCalculating loss..\n")

    l, (clf, bb, obj) = loss(
        (out_clf, true_clf), (out_bb, true_bb), (out_obj, true_obj)
    )

    print(
        f"clf_loss={clf.item(): <14.2f} bb_loss{bb.item(): <14.2f} obj_loss{obj.item(): <14.2f}"
    )

    print(f"loss={l:.2f}")
