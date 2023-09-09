from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import wandb
from src.data.dataset import StanfordDogsDataset
from src.model.label import LabelMap
from src.model.model import StanfordDogsModel


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

    model = StanfordDogsModel(img_size, len(dataset.label_map))
    model = torch.compile(model)
    model = model.to(device)

    epochs = 10
    lr = 1e-3

    n = 1
    run_p = log_dir / f"run-{n}"
    while run_p.exists():
        run_p = log_dir / f"run-{n}"
        n += 1
    run_p.mkdir(exist_ok=True)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="OC5",
        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "architecture": type(model.model).__name__,
            "dataset": "https://huggingface.co/datasets/Alanox/stanford-dogs",
            "epochs": epochs,
        },
    )

    # =============== Run loop
    for epoch in trange(epochs, position=0, desc="Epoch", colour="blue", leave=True):
        losses: list[int] = []
        y_preds: list[str] = []
        y_trues: list[str] = []

        # TRAIN
        for i, datapoints in tqdm(
            enumerate(train_dataloader, start=1),
            desc="Training...",
            position=1,
            leave=False,
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

        log_metrics(
            {
                "val_loss": np.mean(losses),
                "val_acc": accuracy_score(y_trues, y_preds),
            },
            epoch,
            True,
        )


def log_metrics(metrics, epoch, val=False):
    wandb.log(metrics, step=epoch)

    if val:
        tqdm.write(
            f"{epoch=:<4}"
            + " ".join([f"{name}={val:.4f}" for name, val in metrics.items()])
        )


if __name__ == "__main__":
    train()
