from torchmetrics import MeanAbsoluteError, MeanSquaredError, Metric, MetricCollection
from torchmetrics.detection.iou import IntersectionOverUnion


class BoxMetrics(Metric):
    def __init__(self):
        super().__init__()

        self.regression = MetricCollection(
            *[metric() for metric in [MeanSquaredError, MeanAbsoluteError]]
        )

        self.iou = IntersectionOverUnion()

    def update(self, preds, targets):
        pred_boxes = preds["boxes"]
        true_boxes = targets["boxes"]

        self.regression.update(pred_boxes, true_boxes)
        self.iou.update([preds], [targets])

    def compute(self):
        return {**self.regression.compute(), **self.iou.compute()}
