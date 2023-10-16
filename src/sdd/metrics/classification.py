from torchmetrics import Accuracy, F1Score, Metric, MetricCollection, Precision, Recall


class ClassificationMetrics(Metric):
    def __init__(self, num_classes):
        super().__init__()

        classic = [Accuracy, Recall, Precision, F1Score]

        averages = ["macro", "micro"]

        top_ks = [1, 2]

        # self.metrics = MetricCollection(
        #     {
        #         f"{metric.__name__}/{avg}/top{top_k}": metric(
        #             task="multiclass",
        #             num_classes=num_classes,
        #             average=avg,
        #             top_k=top_k,
        #         )
        #         for metric in classic
        #         for avg in averages
        #         for top_k in top_ks
        #     }
        # )
        self.metrics = MetricCollection(
            *[
                metric(
                    task="multiclass",
                    num_classes=num_classes,
                    average="macro",
                )
                for metric in classic
            ]
        )
        # self.metrics = MetricCollection(
        #     *[
        #         MetricCollection(
        #             *[
        #                 MetricCollection(
        #                     *[
        #                         metric(
        #                             task="multiclass",
        #                             num_classes=num_classes,
        #                             average=avg,
        #                             top_k=top_k,
        #                         )
        #                         for metric in classic
        #                     ],
        #                     prefix=avg
        #                 )
        #                 for avg in averages
        #             ],
        #             prefix=str(top_k)
        #         )
        #         for top_k in top_ks
        #     ]
        # )

    def update(self, preds, targets):
        self.metrics.update(preds, targets)

    def compute(self):
        return self.metrics.compute()
