import os
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from ray import air, tune
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import wandb
from src.data.dataset import StanfordDogsDataset
from src.model.label import LabelMap
from src.model.model import StanfordDogsModel

os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
torch.cuda.init()
ray.init()


def train():
    log_dir = Path(__file__).parents[2] / "logs"
    log_dir.mkdir(exist_ok=True)

    img_size = 128

    dataset = StanfordDogsDataset(
        base_transforms=T.Resize((img_size, img_size), antialias=True)
    )

    idx_train, idx_test = train_test_split(
        range(len(dataset)), stratify=dataset.dataset["target"], random_state=0
    )

    # idx_train = idx_train[:1000]
    # idx_test = idx_test[:200]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 16

    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=idx_train)

    def run(config):
        objective = 0

        model = StanfordDogsModel(img_size, len(dataset.label_map))
        model = torch.compile(model)
        model = model.to(device)

        n = 1
        run_p = log_dir / f"run-{n}"
        while run_p.exists():
            run_p = log_dir / f"run-{n}"
            n += 1
        run_p.mkdir(exist_ok=True)

        # start a new wandb run to track this script
        run = wandb.init(
            # set the wandb project where this run will be logged
            project="OC5",
            # track hyperparameters and run metadata
            config={
                "architecture": type(model.model).__name__,
                "dataset": "https://huggingface.co/datasets/Alanox/stanford-dogs",
                **config,
            },
        )
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
        # =============== Run loop
        for epoch in trange(
            config["epochs"],
            position=0,
            desc="Epoch",
            colour="blue",
            leave=True,
            disable=not config["log"],
        ):
            losses: list[int] = []
            y_preds: list[str] = []
            y_trues: list[str] = []

            # TRAIN
            for i, datapoints in tqdm(
                enumerate(train_dataloader, start=1),
                desc="Training...",
                position=1,
                leave=False,
                disable=not config["log"],
            ):
                target = datapoints["target"]
                image = datapoints["image"]
                targets = torch.tensor(dataset.label_map[target]).to(device)

                model.zero_grad()

                output = model(image.to(device))

                loss = F.cross_entropy(output, targets)
                loss.backward()

                # Train log
                with torch.no_grad():
                    preds = F.softmax(output, dim=1)
                    preds = preds.argmax(dim=1)

                    losses.append(loss.item())
                    y_trues.extend(target)
                    y_preds.extend([dataset.label_map[pred.item()] for pred in preds])

                optimizer.step()

            log_metrics(
                {
                    "train_loss": np.mean(losses),
                    "train_acc": accuracy_score(y_trues, y_preds),
                },
                epoch,
                run,
            )
            losses = []
            y_preds = []
            y_trues = []

            # EVAL
            with torch.no_grad():
                for idx in tqdm(
                    idx_test,
                    desc="Evaluation...",
                    colour="magenta",
                    position=1,
                    leave=False,
                    disable=not config["log"],
                ):
                    datapoint = dataset[idx]

                    target = datapoint["target"]
                    image = datapoint["image"].unsqueeze(dim=0)
                    targets = torch.tensor(dataset.label_map[target]).to(device)

                    output = model(image.to(device)).squeeze(dim=0)
                    loss = F.cross_entropy(output, targets)

                    preds = F.softmax(output, dim=0)

                    # Val log
                    losses.append(loss.to("cpu").item())
                    y_preds.append(dataset.label_map[preds.argmax(dim=0).item()])
                    y_trues.append(target)

            pd.DataFrame({"true": y_trues, "preds": y_preds}).to_csv(
                run_p / f"{epoch}-val.csv"
            )

            objective = accuracy_score(y_trues, y_preds)

            log_metrics(
                {
                    "val_loss": np.mean(losses),
                    "val_acc": objective,
                },
                epoch,
                run,
                config["log"],
            )

        run.finish()

        return objective

    scaler = air.ScalingConfig(
        {"CPU": 0.2, "GPU": 0.25},
        num_workers=2,
        # use_gpu=True,
        # resources_per_worker={"CPU": 1, "GPU": 0.2},
    )
    # run({"epochs": 10, "lr": 1e-3, "log": True})
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(run),
            resources=scaler,
        ),
        param_space={"lr": tune.uniform(1e-5, 1e-2), "epochs": 10, "log": False},
        tune_config=tune.TuneConfig(num_samples=20),
        run_config=air.RunConfig(),
    )
    tuner.fit()


def log_metrics(metrics, epoch, run, val=False):
    run.log(metrics, step=epoch)

    if val:
        tqdm.write(
            f"{epoch=:<4}"
            + " ".join([f"{name}={val:.4f}" for name, val in metrics.items()])
        )


if __name__ == "__main__":
    train()
