import os
from pathlib import Path
from pprint import pprint

import ray
import torch
import torch.optim as optim
import torchvision.transforms as T
from ray import air, train, tune
from ray.tune.schedulers import PopulationBasedTraining
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchmo import ConvObserver

import wandb
from sdd.data.collator import BB_Collator
from sdd.data.dataset import StanfordDogsDataset
from sdd.model.loss import ObjectDetectionLoss
from sdd.model.model import StanfordDogsModel
from sdd.model.stepper import Loader, StanfordDogsStepper
from sdd.ray.resolver import keep_tunes, resolve_config
from sdd.utils.dict import deepupdate

torch.cuda.init()

default_config = {
    "architecture": "test",
    "epochs": 10,
    "log": True,
    "img_size": 32,
    "n_cls": -1,
    "batch_size": 8,
    "box_max_amount": 6,
    "wandb": True,
    "true_color": "blue",
    "out_color": "red",
    "optimizer": {"name": "AdamW", "betas": (0.9, 0.999), "weight_decay": 1e-2},
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
    "ray": {
        "batch_size": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
        "lr": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
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

    log_dir = Path(__file__).parents[3] / "logs"
    log_dir.mkdir(exist_ok=True)

    def run(config):
        objective = 0

        config["real_batch_size"] = config.get("real_batch_size", config["batch_size"])

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Config params
        box_max_amount = config["box_max_amount"]
        img_size = config["img_size"]
        n_cls = config.pop("n_cls", -1)

        dataset = StanfordDogsDataset(img_size, num_classes=n_cls)

        # Extract other config params
        model_name = config["architecture"]
        batch_size = config["batch_size"]
        log = config["log"]
        config["num_classes"] = dataset.num_classes_
        epochs = config["epochs"]
        optimizer_params = config["optimizer"]
        group = config.get("group", None)

        batch_ratio = config["batch_size"] / config["real_batch_size"]
        assert int(batch_ratio) == float(batch_ratio)

        # Relative lr - Stable loss
        # Ref batch_size = 8
        # k = batch_size / 8
        # lr * sqrt(k)
        # lr_ratio = (batch_size * config["lr"]) / 128
        # config["lr_ratio"] = lr_ratio

        lr = config["lr"]
        wandb_active = config["wandb"]
        decay = config.get("decay", None)

        idx_train, idx_test = train_test_split(
            range(len(dataset)), stratify=dataset.dataset["target"], random_state=0
        )

        collator = BB_Collator(dataset.label_map, box_max_amount)

        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=idx_train,
            collate_fn=collator,
        )

        val_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=idx_test,
            collate_fn=collator,
        )

        model = StanfordDogsModel(
            img_size, len(dataset.label_map), box_max_amount, model_name=model_name
        ).to(device)

        n = 1
        run_p = log_dir / f"run-{n}"
        while run_p.exists():
            run_p = log_dir / f"run-{n}"
            n += 1
        run_p.mkdir(exist_ok=True)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=optimizer_params["betas"],
            weight_decay=optimizer_params["weight_decay"],
        )

        scheduler = None
        if decay is not None:
            scheduler = StepLR(
                optimizer,
                step_size=decay.get("step_size", 50),
                gamma=decay.get("gamma", 0.99),
            )
            config["scheduler"] = type(scheduler).__name__

        loss = ObjectDetectionLoss(img_size)

        # Start a new wandb run to track this script
        run = wandb.init(
            project="SDD",
            config={
                "dataset": "https://huggingface.co/datasets/Alanox/stanford-dogs",
                **config,
                "optimizer": type(optimizer).__name__,
            },
            tags=[group] if group is not None else None,
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

    if tuning:
        os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
        os.environ["WANDB_SILENT"] = "true"

        ray.init()
        scaler = air.ScalingConfig({"CPU": 0.2, "GPU": 0.25}, num_workers=2)

        param_space_config = resolve_config(final_config)
        param_tune_config = keep_tunes(param_space_config)

        print("param_space_config :")
        pprint(param_space_config)
        print("param_tune_config : ")
        pprint(param_tune_config)

        perturbation_interval = 5
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=perturbation_interval,
            metric="val_MulticlassAccuracy",
            mode="max",
            hyperparam_mutations=param_tune_config,
        )

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(run),
                resources=scaler,
            ),
            run_config=train.RunConfig(
                checkpoint_config=train.CheckpointConfig(num_to_keep=2)
            ),
            tune_config=tune.TuneConfig(scheduler=scheduler, num_samples=2),
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
