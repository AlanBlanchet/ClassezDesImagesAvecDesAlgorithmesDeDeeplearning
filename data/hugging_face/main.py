"""
This script cleans the Stanford Dogs dataset
"""

import re
import tarfile
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":
    hugging_face_p = Path(__file__).parent.resolve()
    raw_p = hugging_face_p / "raw"
    export_p = hugging_face_p / "stanford-dogs"

    export_p.mkdir(exist_ok=True)

    images_p = raw_p / "images"
    annotations_p = raw_p / "annotations"

    df = pd.DataFrame(columns=["name", "annotations", "target"])

    box_values = ["xmin", "ymin", "xmax", "ymax"]

    with tarfile.open(export_p / "images.tar.gz", "w:gz") as tar:
        for dog_dir in tqdm(list(images_p.iterdir())):
            if not dog_dir.is_dir():
                continue

            name = re.sub(r"^.*?-", "", dog_dir.name)
            name = [
                s.capitalize()
                for s in name.replace("_", " ").replace("-", " ").split(" ")
            ]
            name = " ".join(name)

            for dog_img in dog_dir.iterdir():
                # Fix jpg
                Image.open(dog_img).convert("RGB").save(dog_img)

                annotations: list[list[int]] = []
                # Parse annotations
                with open(annotations_p / dog_dir.name / dog_img.stem) as f:
                    objects_t = BeautifulSoup(f.read(), "xml").find_all("object")

                    for object_t in objects_t:
                        annotations.append(
                            [
                                int(object_t.find(box_value).getText())
                                for box_value in box_values
                            ]
                        )

                df.loc[len(df)] = [dog_img.name, annotations, name]

            tar.add(str(dog_dir), arcname="images")

    df.to_csv(export_p / "metadata.csv", index=False)
