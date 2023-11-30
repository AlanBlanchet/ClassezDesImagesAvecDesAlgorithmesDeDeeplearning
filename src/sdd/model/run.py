import math
import os
from pathlib import Path
from pprint import pprint

import ray
import torch
import torch.optim as optim
import yaml
from ray import air, train, tune
from ray.tune.schedulers import ASHAScheduler
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.utils.data import DataLoader

import wandb
from sdd.compat.utils import chose_if_task_bb, is_task_bb
from sdd.data.augmentations import BaseAugmentations, StanfordDogsAugmentations
from sdd.data.collator import BB_Collator
from sdd.data.dataset import StanfordDogsDataset
from sdd.model.loss import ObjectClassificationLoss, ObjectDetectionLoss, get_loss
from sdd.model.model import StanfordDogsModel
from sdd.model.stepper import Loader, StanfordDogsStepper
from sdd.optimizer import Lion
from sdd.ray.resolver import resolve_config
from sdd.utils.dict import deepupdate
from sdd.utils.dir import next_run_dir

torch.cuda.init()

RUN_DIR = (Path(__file__).parents[3] / "runs").absolute()
RUN_DIR.mkdir(exist_ok=True)

default_config = {
    "architecture": "test",
    "epochs": 10,
    "log": True,
    "img_size": 32,
    "n_cls": -1,
    "batch_size": 8,
    "box_max_amount": 8,
    "wandb": True,
    "true_color": "blue",
    "out_color": "red",
    "optimizer": {"name": "AdamW", "betas": (0.9, 0.999), "weight_decay": 1e-2},
    "run_p": str(RUN_DIR),
    "mosaic": False,
    "loss": None,
}

default_ray_names = [
    "architecture",
    "epochs",
    "img_size",
    "n_cls",
    "box_max_amount",
    "wandb",
    "true_color",
    "out_color",
    "optimizer",
]

default_ray = {
    **{name: default_config[name] for name in default_ray_names},
    "log": False,
    "log_p": str(next_run_dir(RUN_DIR, create=False)),
    "ray": {
        "batch_size": {
            "type": "choice",
            # "value": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
            "value": [4, 8, 16, 32, 64],
        },
        "lr": {"type": "loguniform", "value": [1e-2, 1e-6]},
    },
}


def start(global_config: dict):
    tuning_config = global_config.get("ray", False)
    tuning = False

    if type(tuning_config) == bool and tuning_config:
        global_config["ray"] = {}
        tuning = True

    default = default_ray if tuning else default_config
    final_config = deepupdate({}, default, global_config)

    log_dir = Path(final_config["run_p"])

    def run(config):
        config["real_batch_size"] = config.get("real_batch_size", config["batch_size"])

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Config params
        task = config.get("task", "detection")
        format = config.get("format", "pascal_voc")
        box_max_amount = config["box_max_amount"] if is_task_bb(task) else 1
        img_size = config["img_size"]
        n_cls = config.pop("n_cls", -1)

        # Extract other config params
        model_name = config["architecture"]
        batch_size = config["batch_size"]
        log = config["log"]
        epochs = config["epochs"]
        loss_name = config["loss"]
        optimizer_params = deepupdate(
            {}, default_config["optimizer"], config["optimizer"]
        )
        group = config.get("group", None)

        batch_ratio = config["batch_size"] / config["real_batch_size"]
        assert int(batch_ratio) == float(batch_ratio)
        lr = config["lr"]
        wandb_active = config["wandb"]
        decay = config.get("decay", None)
        min_lr = config.get("min_lr", None)
        mosaic = config["mosaic"]
        optimizer_name = optimizer_params["name"]

        groups = []
        if group is not None:
            groups.append(group)
        groups.append(task)

        augmentations = StanfordDogsAugmentations(config, format)
        dataset = StanfordDogsDataset(
            config,
            img_size,
            num_classes=n_cls,
            augmentations=augmentations,
            mosaic=mosaic,
        )
        config["num_classes"] = dataset.num_classes_

        collator = BB_Collator(dataset.label_map, box_max_amount)

        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=dataset.train_idx_,
            collate_fn=collator,
        )

        val_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=dataset.val_idx_,
            collate_fn=collator,
        )

        model = StanfordDogsModel(
            config,
            img_size,
            len(dataset.label_map),
            box_max_amount,
            model_name=model_name,
        ).to(device)

        # Run dir
        run_p = next_run_dir(log_dir)
        (run_p / "config.yml").write_text(yaml.dump(config))
        (run_p / "label2id.yml").write_text(yaml.dump(dataset.label_map._label2id))

        optimizer = None
        if optimizer_name == "AdamW":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=optimizer_params["betas"],
                weight_decay=optimizer_params["weight_decay"],
            )
        elif optimizer_name == "Lion":
            optimizer = Lion(
                model.parameters(),
                lr=lr,
                betas=optimizer_params["betas"],
                weight_decay=optimizer_params["weight_decay"],
            )

        # Initialize trained or stopped
        loaded_checkpoint = train.get_checkpoint()
        if loaded_checkpoint:
            pprint(config)

            raise NotImplementedError()
            with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
                model_state, optimizer_state = torch.load(
                    os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
                )
                model.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)

        scheduler = None
        if decay is not None:
            scheduler = StepLR(
                optimizer,
                step_size=decay.get("step_size", 50),
                gamma=decay.get("gamma", 0.99),
            )
        elif min_lr is not None:
            nb_iters = epochs * math.ceil(len(train_dataloader) / batch_ratio)
            gamma = (min_lr / lr) ** (1 / nb_iters)

            scheduler = ExponentialLR(optimizer, gamma=gamma)

        config["scheduler"] = type(scheduler).__name__

        loss = get_loss(loss_name, task, img_size, format)

        # Start a new wandb run to track this script
        run = wandb.init(
            project="SDD",
            config={
                "dataset": "https://huggingface.co/datasets/Alanox/stanford-dogs",
                **config,
                "optimizer": type(optimizer).__name__,
            },
            tags=groups,
            mode=None if wandb_active else "disabled",
        )

        if log:
            pprint(config)

        stepper = StanfordDogsStepper(
            config,
            device,
            [
                Loader(train_dataloader, "train"),
                Loader(val_dataloader, "val"),
            ],
            dataset,
            model,
            loss,
            optimizer,
            scheduler,
            log,
            out_p=run_p,
            wandb_run=run,
        )

        # =============== Run loop
        stepper.run(epochs)

        # Save model
        torch.save(model.state_dict(), (run_p / "checkpoint.pt"))

    if tuning:
        os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
        os.environ["WANDB_SILENT"] = "true"
        os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

        ray.init()
        scaler = air.ScalingConfig({"CPU": 0.2, "GPU": 0.25}, num_workers=2)

        param_space_config = resolve_config(final_config)
        # param_tune_config = keep_tunes(param_space_config)

        print("param_space_config :")
        pprint(param_space_config)
        # print("param_tune_config :")
        # pprint(param_tune_config)

        # perturbation_interval = 2
        # pbt = PopulationBasedTraining(
        #     time_attr="training_iteration",
        #     perturbation_interval=perturbation_interval,
        #     mode="max",
        #     hyperparam_mutations=param_tune_config,
        #     log_config=False,
        # )
        scheduler = ASHAScheduler(time_attr="training_iteration", mode="max")

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(run),
                resources=scaler,
            ),
            run_config=train.RunConfig(
                checkpoint_config=train.CheckpointConfig(),
            ),
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=100,
                max_concurrent_trials=4,
                metric="val_MulticlassAccuracy",
            ),
            param_space=param_space_config,
        )
        tuner.fit()
    else:
        run(final_config)


if __name__ == "__main__":
    config = {
        "architecture": "vgg16",
        "epochs": 5,
        "lr": 6e-3,
        "log": True,
        "img_size": 512,
        "n_cls": 5,
        "batch_size": 32,
    }

    start(config)
