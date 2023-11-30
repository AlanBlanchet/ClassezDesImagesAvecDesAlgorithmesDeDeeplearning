from click import argument, command


@command()
@argument("run", type=str)
@argument("folder", type=str)
@argument("out", type=str)
def main(run: str, folder: str, out: str):
    import shutil
    from pathlib import Path

    import pandas as pd
    import torchvision.transforms.functional as TF
    from PIL import Image

    from sdd.model.model import StanfordDogsModel
    from sdd.utils.configs import load_yml

    run_p = Path(run)
    folder_p = Path(folder)
    out_p = Path(out)

    config = load_yml(run_p / "config.yml")
    model_name = config["architecture"]

    label2id: dict[str, int] = load_yml(run_p / "label2id.yml")
    id2label = {v: k for k, v in label2id.items()}

    img_size = config.get("img_size", 224)

    model = StanfordDogsModel(
        config,
        img_size,
        len(label2id),
        1,  # No boxes for classification
        model_name=model_name,
    )

    out_p.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame({"file": [], "prediction": []})
    for file_p in folder_p.iterdir():
        image = Image.open(file_p)
        image = image.resize((img_size, img_size))

        image = TF.pil_to_tensor(image) / 255

        out_pred = model(image.unsqueeze(dim=0)).squeeze(dim=0)

        idx = out_pred.softmax(dim=0).argmax(dim=0).item()

        dog = id2label[idx]

        df.loc[len(df)] = [str(file_p), dog]

        dog_p = out_p / dog
        dog_p.mkdir(exist_ok=True)
        shutil.copy(file_p, dog_p)

    df.to_csv(out_p / "predictions.csv", index=False)
