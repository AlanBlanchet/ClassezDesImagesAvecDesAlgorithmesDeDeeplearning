from pathlib import Path


def next_run_dir(log_dir: Path):
    n = 1
    run_p = log_dir / f"run-{n}"
    while run_p.exists():
        run_p = log_dir / f"run-{n}"
        n += 1
    run_p.mkdir(exist_ok=True)

    return run_p
