from torchmetrics import MeanAbsoluteError, MeanSquaredError, Metric, MetricCollection
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision

box_format_map = {"pascal_voc": "xyxy", "coco": "xywh", "yolo": "cxcywh"}


class BoxMetrics(Metric):
    def __init__(self, format):
        super().__init__()

        box_format = box_format_map[format]

        self.regression = MetricCollection(
            [metric() for metric in [MeanSquaredError, MeanAbsoluteError]],
            compute_groups=False,
        )

        self.iou = IntersectionOverUnion(box_format=box_format)
        self.map = MetricCollection(
            {
                f"mAP/{average}": MeanAveragePrecision(
                    box_format=box_format, average=average
                )
                for average in ["micro", "macro"]
            },
            compute_groups=False,
        )

    def update(self, preds, targets):
        pred_bbs_b, pred_clfs_b, pred_obj_b = preds
        true_bbs_b, true_clfs_b, true_obj_b = targets

        iou_list = [[], []]  # [pred, true]
        map_list = [[], []]

        for B in range(len(pred_bbs_b)):
            # Each image
            pred_bbs = pred_bbs_b[B]
            pred_clfs = pred_clfs_b[B]
            pred_obj = pred_obj_b[B]

            true_bbs = true_bbs_b[B]
            true_clfs = true_clfs_b[B]
            true_obj = true_obj_b[B]

            mask = true_obj == 1

            self.regression.update(pred_bbs[mask], true_bbs[mask])

            # print("=" * 20)
            # print(f"{pred_bbs.shape=}")
            # print(f"{pred_clfs.shape=}")
            # print(f"{pred_obj.shape=}")
            # print(f"{true_bbs.shape=}")
            # print(f"{true_clfs.shape=}")
            # print(f"{true_obj.shape=}")
            # print(f"{mask=}")
            # print(f"{pred_bbs[mask].shape=}")
            # print(f"{pred_clfs[mask].shape=}")
            # print(f"{pred_obj[mask].shape=}")
            # print(f"{true_bbs[mask].shape=}")
            # print(f"{true_clfs[mask].shape=}")
            # print(f"{true_obj[mask].shape=}")

            # IoU
            iou_list[0].append(
                {
                    "boxes": pred_bbs[mask],
                    "labels": pred_clfs[mask].softmax(dim=1).argmax(dim=1),
                }
            )
            iou_list[1].append(
                {
                    "boxes": true_bbs[mask],
                    "labels": true_clfs[mask],
                }
            )

            # mAP
            map_list[0].append(
                {
                    "boxes": pred_bbs,
                    "labels": pred_clfs.softmax(dim=1).argmax(dim=1),
                    "scores": pred_obj,
                }
            )
            map_list[1].append({"boxes": true_bbs, "labels": true_clfs})

        self.iou.update(*iou_list)
        self.map.update(*map_list)

    def compute(self):
        mAP = self.map.compute()

        return {
            **self.regression.compute(),
            **self.iou.compute(),
            **{
                f"mAP/{average}": mAP[f"mAP/{average}_map"]
                for average in ["micro", "macro"]
            },
        }
