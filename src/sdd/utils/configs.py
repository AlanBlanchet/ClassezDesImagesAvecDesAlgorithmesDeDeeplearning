from itertools import takewhile
from pathlib import Path

import inquirer
import yaml

root_p = (Path(__file__).parents[3] / "configs").resolve()


def path_until(path: Path, until: str = root_p.name):
    split_paths = list(
        reversed(list(takewhile(lambda p: p != until, reversed(str(path).split("/")))))
    )

    return ("/".join(split_paths)).strip("/")


def parse(path: Path) -> list[Path]:
    if path is not None:
        if path.is_dir():
            return list(path.rglob("*.yml"))
        else:
            return [path]
    else:
        path = root_p

        all_configs_p = list(path.rglob("*.yml"))

        questions = inquirer.Checkbox(
            "configs",
            message="Chose the configs you want to launch (Press <space> to select, Enter when finished).",
            choices=[path_until(c) for c in all_configs_p],
        )

        confs = inquirer.prompt([questions])["configs"]

        return [root_p / c for c in confs]


def load_yml(path: Path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config
