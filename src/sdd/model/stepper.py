from pathlib import Path
from pprint import pprint
from typing import Literal, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.utils as U
from attrs import define
from ray import train
from ray.train import Checkpoint
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from tqdm import tqdm, trange
from wandb.wandb_run import Run

from sdd.compat.utils import is_task_bb, return_if_task_bb
from sdd.data.maps.basic import rescale
from sdd.data.standard import StandardStanfordDogsDataset
from sdd.metrics.box import BoxMetrics
from sdd.metrics.classification import ClassificationMetrics
from sdd.model.model import StanfordDogsModel
from sdd.utils.dict import deep_dict_parse_tensor


class Loader(NamedTuple):
    dataloader: DataLoader
    type: Literal["train", "val", "test"]


@define(slots=False)
class StanfordDogsStepper:
    config: dict
    device: str
    dataloaders: list[Loader]
    dataset: StandardStanfordDogsDataset
    model: StanfordDogsModel
    loss: nn.Module
    optimizer: nn.Module
    scheduler: nn.Module | None
    log: bool
    out_p: Path
    wandb_run: Run

    rand_item = 0

    _fit = []

    _epoch = 0

    def __attrs_post_init__(self):
        self.task = self.config.get("task", "detection")

        self._clf_metrics = ClassificationMetrics(self.dataset.num_classes_).to(
            self.device
        )

        if is_task_bb(self.task):
            self._bb_metrics = BoxMetrics().to(self.device)
            self._bb_loss_metric = MeanMetric().to(self.device)
            self._obj_loss_metric = MeanMetric().to(self.device)

        self._loss_metric = MeanMetric().to(self.device)
        self._clf_loss_metric = MeanMetric().to(self.device)

        self.examples_p = self.out_p / "examples"
        self.examples_p.mkdir(exist_ok=True)

    def reset(self):
        self._clf_metrics.reset()
        if is_task_bb(self.task):
            self._bb_loss_metric.reset()
            self._obj_loss_metric.reset()
        self._clf_loss_metric.reset()
        self._loss_metric.reset()

    def train(self, training: bool = True):
        if self.model.training != training:
            self.model.train(training)
            if training:
                self._fit = []
            self.reset()

    def eval(self):
        self.train(False)

    def _format_metrics(self, metrics: dict) -> dict:
        prefix = "train" if self.model.training else "val"
        return {f"{prefix}_{k}": v for k, v in metrics.items()}

    def metrics(self) -> dict:
        addon = {}

        loss = self._loss_metric.compute()
        self._fit.append(loss)

        if not self.model.training:
            addon = {"fit": self._fit[0] / self._fit[1]}

        bb_metrics = {}
        if is_task_bb(self.task):
            bb_metrics = {
                **self._bb_metrics.compute(),
                "bb_loss": self._bb_loss_metric.compute(),
                "obj_loss": self._obj_loss_metric.compute(),
            }

        return {
            **self._format_metrics(
                {
                    **self._clf_metrics.compute(),
                    "clf_loss": self._clf_loss_metric.compute(),
                    "loss": loss,
                    **bb_metrics,
                }
            ),
            **addon,
        }

    def run(self, epochs: int):
        # Val samples
        samples = self.dataloaders[-1].dataloader.batch_sampler.sampler

        # Epoch cycles
        for epoch in trange(
            epochs,
            position=0,
            desc="Epoch",
            colour="blue",
            leave=True,
            disable=not self.log,
        ):
            self._epoch = epoch
            self.rand_item = np.random.choice(samples, 1)[0]

            # Dataloaders iteration
            for dataloader, name in self.dataloaders:
                is_train = name == "train"
                self.train(is_train)

                with torch.set_grad_enabled(is_train):
                    # Train / Val
                    for i, datapoints in tqdm(
                        enumerate(dataloader, start=1),
                        desc=f"{name}...",
                        colour="magenta",
                        position=1,
                        leave=False,
                        disable=not self.log,
                    ):
                        self(i, datapoints)

                    # Metrics
                    addon = {}
                    if not is_train and self.scheduler:
                        addon = {"lr": self.scheduler.get_last_lr()[0]}

                    metrics = {**self.metrics(), **addon}

                    self.log_metrics(metrics, epoch, not is_train)

            if (
                epoch > 0
                and epoch % 1 == 0
                and not is_train
                and self.config.get("ray", False)
            ):
                checkpoints_p = self.out_p / "checkpoints"
                checkpoints_p.mkdir(exist_ok=True)

                checkpoint_p = checkpoints_p / f"{self._epoch}"
                checkpoint_p.mkdir(exist_ok=True)

                torch.save(
                    (self.model.state_dict(), self.optimizer.state_dict()),
                    checkpoint_p / "checkpoint.pt",
                )
                checkpoint = Checkpoint.from_directory(checkpoint_p)

                train.report(
                    {"checkpoint": checkpoint, **deep_dict_parse_tensor(metrics)},
                )

        self.wandb_run.finish()

    def __call__(self, idx: int, datapoints):
        training = self.model.training

        # Data collecting
        image = datapoints["image"].to(self.device)
        true_clf = datapoints["target"].to(self.device)
        true_bb = datapoints["annotations"].to(self.device)
        true_obj = datapoints["annotations_mask"].to(self.device)

        # Forward
        if is_task_bb(self.task):
            out_clf, out_bbs, out_obj = self.model(image)

            # Loss
            loss_sum, (clf_loss, bb_loss, obj_loss) = self.loss(
                (out_clf, true_clf.long()), (out_bbs, true_bb), (out_obj, true_obj)
            )
        else:
            true_clf = true_clf[..., 0]

            out_clf = self.model(image)
            clf_loss = loss_sum = self.loss((out_clf, true_clf.long()))

        # bb_loss = (
        #     F.smooth_l1_loss(out_bbs, true_bb, reduction="none").sum(dim=-1)
        #     * true_obj
        # )  # (B,N)
        # bb_loss = (bb_loss.sum(dimrun=-1) / true_obj.sum(dim=-1)).mean(
        #     dim=0
        # )

        if training:
            # Zero gradient
            self.model.zero_grad()

            # Backpropagation
            loss_sum.backward()

        # Log
        with torch.no_grad():
            if is_task_bb(self.task):
                mask = true_obj == 1

                out_clf_masked = out_clf[mask]
                true_clf_masked = true_clf[mask]

                probas_clf = F.softmax(out_clf_masked, dim=1)

                preds_bbs = out_bbs[mask]

                self._clf_metrics.update(out_clf_masked, true_clf_masked)

                preds_clf = probas_clf.argmax(dim=1)

                self._bb_metrics.update(
                    {"boxes": preds_bbs, "labels": preds_clf},
                    {"boxes": true_bb[mask], "labels": true_clf_masked},
                )

                self._bb_loss_metric.update(bb_loss)
                self._obj_loss_metric.update(obj_loss)
            else:
                probas_clf = F.softmax(out_clf, dim=1)

                self._clf_metrics.update(probas_clf, true_clf)

            self._clf_loss_metric.update(clf_loss)
            self._loss_metric.update(loss_sum)

            indexes = datapoints["index"]
            rand = self.rand_item

            # Pick a random sample for logging
            if not training and rand in indexes and is_task_bb(self.task):
                rand_idx_pos = (indexes == rand).nonzero().squeeze(dim=0)
                batch_idx = rand_idx_pos.item()

                epoch_p = self.examples_p / f"{self._epoch}"
                epoch_p.mkdir(exist_ok=True)

                show_size = 1024

                rescale_size = (show_size,) * 2

                true_mask = mask[batch_idx]
                true_clfs = true_clf[batch_idx, true_mask]
                bbs = rescale(true_bb[batch_idx, true_mask], rescale_size)

                rand_image = TF.resize(
                    (image[batch_idx] * 255).to(torch.uint8), rescale_size
                )
                rand_obj = out_obj[batch_idx].sigmoid()
                rand_mask = rand_obj > 0.5

                rand_pred_clf = F.softmax(out_clf[batch_idx, rand_mask], dim=1)
                rand_clf = rand_pred_clf.argmax(dim=1)
                rand_bbs = out_bbs[batch_idx, rand_mask]
                rand_bbs = rescale(torch.clip(rand_bbs, 0, 1), rescale_size)
                for bb in rand_bbs:
                    if bb[0] > bb[2]:
                        bb[[0, 2]] = bb[[2, 0]]
                    if bb[1] > bb[3]:
                        bb[[1, 3]] = bb[[3, 1]]

                # observer = ConvObserver(model, limit=5)
                # observer(image)
                # observer.save_figs(epoch_p)

                # model.to(device)
                # image.to(device)

                # Build predicted image
                # img: torch.Tensor = image[batch_idx]
                # img = (img * 255).type(torch.uint8)

                # Current
                rand_image = U.draw_bounding_boxes(
                    rand_image,
                    rand_bbs.type(torch.int32),
                    colors=self.config["out_color"],
                    labels=self.dataset.label_map[rand_clf],
                )
                rand_image = U.draw_bounding_boxes(
                    rand_image,
                    bbs.type(torch.int32),
                    colors=self.config["true_color"],
                    labels=self.dataset.label_map[true_clfs],
                )

                TF.to_pil_image(rand_image).save(epoch_p / f"{rand}.jpg")

                # Original
                self.dataset.original(True)
                original_item = self.dataset[rand]
                img = original_item["original_image"].type(torch.uint8).permute(2, 0, 1)
                img = TF.resize(img, rescale_size)
                # annots = original_item["original_annotations"]
                self.dataset.original(False)

                img = U.draw_bounding_boxes(
                    img,
                    rand_bbs.type(torch.int32),
                    colors=self.config["out_color"],
                    labels=self.dataset.label_map[rand_clf],
                )
                img = U.draw_bounding_boxes(
                    img,
                    bbs.type(torch.int32),
                    colors=self.config["true_color"],
                    labels=self.dataset.label_map[true_clfs],
                )

                TF.to_pil_image(img).save(epoch_p / f"{rand}_original.jpg")

        if training:
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

    def log_metrics(self, metrics, epoch, val=False):
        self.wandb_run.log(metrics, step=epoch)

        if val and self.log:
            tqdm.write(
                f"{epoch=:<4}"
                + " ".join(
                    [self.format_metric(name, val) for name, val in metrics.items()]
                )
            )

    def format_metric(self, key, val):
        if key == "lr":
            return f"{key}={val:.8f}"
        return f"{key}={val:.4f}"
