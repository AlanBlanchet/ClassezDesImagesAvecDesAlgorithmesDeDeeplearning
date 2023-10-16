from pathlib import Path

from click import command, option


@command("sdd")
@option("--config", type=Path, default=None)
def main(config=None):
    from sdd.utils import load_yml, parse, path_until, root_p

    config = parse(config)

    import traceback

    from tqdm import tqdm

    from sdd.model.run import start

    for config_p in tqdm(config, disable=len(config) <= 1):
        try:
            conf = load_yml(config_p)

            conf["name"] = path_until(config_p)

            start(conf)
        except Exception as e:
            print(f"Error occured for config {config_p}")
            traceback.print_exc()
