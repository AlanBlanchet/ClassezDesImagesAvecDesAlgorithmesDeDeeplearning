import torch
import torch.nn as nn

from sdd.compat.utils import is_task_bb


class HF(nn.Module):
    def __init__(self, model, task):
        super().__init__()
        self.model = model

        self.task = task

    def forward(self, x):
        out = self.model(pixel_values=x)
        x = out["logits"]

        if is_task_bb(self.task):
            out_bb = out["pred_boxes"][:8]

            out_clf = x[:, :8].squeeze(dim=-1)

            clfs = out_clf.softmax(dim=-1).argmax(dim=-1)

            out_objn = torch.zeros(out_clf.shape, device=out_clf.device, dtype=x.dtype)
            # 0 = N/A
            out_objn[clfs != 0] = out_clf[:, clfs != 0]

            # Bounding boxes coco

            return out_clf, out_bb, out_objn
        return x
