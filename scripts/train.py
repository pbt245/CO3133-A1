#!/usr/bin/env python
"""Train one experiment config with one seed, then evaluate the selected checkpoint on test.

Examples
  python scripts/train.py --config configs/fashion_mnist/cnn.yaml --seed 42
  python scripts/train.py --config configs/fashion_mnist/mlp.yaml \
      --set data.name=mnist data.augment.hflip=false train.epochs=3 experiment.group=debug

Run folder: outputs/runs/<dataset>/<experiment>/seed<seed>/
  config.yaml (resolved) | env.json | train.log | model_architecture.txt | history.csv | curves.png
  train_summary.json | checkpoints/best.pt, last.pt | eval/ (test metrics + figures)
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.engine.evaluate import evaluate_run  # noqa: E402
from src.engine.trainer import run_training  # noqa: E402
from src.utils.config import apply_overrides, load_config, save_config  # noqa: E402
from src.utils.env import environment_info, git_info, resolve_device  # noqa: E402
from src.utils.io import resolve_path, save_json  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None, help="training seed (the data split seed stays fixed)")
    p.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=3")
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:0 | mps | cpu")
    p.add_argument("--no-eval", action="store_true", help="skip test evaluation after training")
    p.add_argument("--overwrite", action="store_true", help="delete an existing run folder first")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    for key in ("data", "model", "train", "optim"):
        if key not in cfg:
            raise ValueError(f"Config is missing section '{key}'")
    if not cfg["data"].get("name") or not cfg["model"].get("name"):
        raise ValueError("data.name and model.name must be set")
    cfg.setdefault("experiment", {})
    exp = cfg["experiment"].get("name") or cfg["model"]["name"]
    cfg["experiment"]["name"] = exp
    seed = int(cfg["seed"])

    run_dir = resolve_path(cfg["output"]["root"]) / cfg["data"]["name"] / exp / f"seed{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        if not args.overwrite:
            print(f"Run folder already exists: {run_dir}\nUse --overwrite to replace it.")
            sys.exit(1)
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{cfg['data']['name']}-{exp}-s{seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    cfg["run"] = {
        "run_id": run_id,
        "config_file": str(Path(args.config).as_posix()),
        "overrides": list(args.set),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    save_config(cfg, run_dir / "config.yaml")

    logger = get_logger(run_id, run_dir / "train.log")
    device = resolve_device(args.device)
    env = environment_info(device)
    env["git"] = git_info()
    save_json(env, run_dir / "env.json")
    logger.info("Run %s | device %s | torch %s | git %s", run_id, device, env["versions"]["torch"], env["git"])
    if env["git"]["commit"] is None:
        logger.warning("Not a git repository: results will not be traceable to a commit.")
    elif env["git"]["dirty"]:
        logger.warning("Uncommitted changes present: commit before final runs for traceability.")

    run_training(cfg, run_dir, device, logger)
    if not args.no_eval:
        evaluate_run(run_dir, device, split="test", logger=logger)
    logger.info("Done: %s", run_dir)


if __name__ == "__main__":
    main()
