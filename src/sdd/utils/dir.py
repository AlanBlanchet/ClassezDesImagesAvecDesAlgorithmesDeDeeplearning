from pathlib import Path


def next_run_dir(log_dir: Path, create=True):
    n = 1
    run_p = log_dir / f"run-{n}"
    while run_p.exists():
        run_p = log_dir / f"run-{n}"
        n += 1

    if create:
        run_p.mkdir(exist_ok=True, parents=True)

    return run_p
