from typing import Any

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics import (
    Accuracy,
    ConfusionMatrix,
    F1Score,
    Metric,
    MetricCollection,
    Precision,
    Recall,
)

from sdd.model.label import LabelMap


class ClassificationMetrics(Metric):
    def __init__(self, label_map: LabelMap):
        super().__init__()

        classic = [Accuracy, Recall, Precision, F1Score]
        averages = ["macro", "micro"]
        top_ks = [1, 2]

        self.label_map = label_map

        self.metrics = MetricCollection(
            {
                f"{metric.__name__}/{avg}/top{top_k}": metric(
                    task="multiclass",
                    num_classes=len(label_map),
                    average=avg,
                    top_k=top_k,
                )
                for metric in classic
                for avg in averages
                for top_k in top_ks
            },
            compute_groups=False,
        )

        self.cm = ConfusionMatrix(
            num_classes=len(label_map), task="multiclass", normalize="true"
        )

    def update(self, preds, targets):
        self.metrics.update(preds, targets)
        self.cm.update(preds, targets)

    def compute(self):
        return self.metrics.compute()

    def plot(self):
        cm = self.cm.compute().cpu()
        cmd = ConfusionMatrixDisplay(
            cm.numpy(), display_labels=self.label_map._id2label.values()
        )
        return {"Confusion_Matrix": lambda: cmd.plot(values_format=".2f").figure_}
