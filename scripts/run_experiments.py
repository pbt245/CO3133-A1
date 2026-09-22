#!/usr/bin/env python
"""Run many experiments sequentially (cross-platform, one subprocess per run).

Examples
  python scripts/run_experiments.py --dataset fashion_mnist --models linear mlp --seeds 42
  python scripts/run_experiments.py --dataset fashion_mnist --group main --seeds 42 123 2026
  python scripts/run_experiments.py --dataset fashion_mnist --group ablation --seeds 42 123 2026
  python scripts/run_experiments.py --dataset cifar10 --group main --seeds 42

Finished runs (eval/test_metrics.json present) are skipped unless --force.
Unfinished run folders are overwritten automatically.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.io import resolve_path

MAIN_ORDER = ["linear", "mlp", "cnn", "lstm", "gru", "transformer"]


def find_configs(dataset: str, group: str):
    d = ROOT / "configs" / dataset
    if not d.exists():
        raise FileNotFoundError(f"No config folder {d}")
    main = sorted((p for p in d.glob("*.yaml") if not p.name.startswith("_")),
                  key=lambda p: (MAIN_ORDER.index(p.stem) if p.stem in MAIN_ORDER else 99, p.stem))
    abl = sorted((d / "ablations").glob("*.yaml")) if (d / "ablations").exists() else []
    return {"main": main, "ablation": abl, "all": main + abl}[group]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["fashion_mnist", "cifar10"])
    p.add_argument("--group", default="main", choices=["main", "ablation", "all"])
    p.add_argument("--models", nargs="*", default=None, help="filter by experiment name")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    configs = find_configs(args.dataset, "all" if args.models else args.group)
    jobs = []
    for cfg_path in configs:
        cfg = load_config(cfg_path)
        exp = cfg["experiment"].get("name") or cfg["model"]["name"]
        if args.models and exp not in args.models:
            continue
        for seed in args.seeds:
            jobs.append((cfg_path, cfg, exp, seed))
    if not jobs:
        print("No matching experiments.")
        return

    results = []
    for i, (cfg_path, cfg, exp, seed) in enumerate(jobs, 1):
        run_dir = resolve_path(cfg["output"]["root"]) / cfg["data"]["name"] / exp / f"seed{seed}"
        tag = f"[{i}/{len(jobs)}] {args.dataset}/{exp}/seed{seed}"
        if (run_dir / "eval" / "test_metrics.json").exists() and not args.force:
            print(f"{tag}: already finished, skipping")
            results.append((tag, "skipped", 0.0))
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(cfg_path),
               "--seed", str(seed), "--device", args.device]
        if run_dir.exists():
            cmd.append("--overwrite")
        print(f"\n{tag}: {' '.join(cmd)}")
        if args.dry_run:
            results.append((tag, "dry-run", 0.0))
            continue
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        results.append((tag, "ok" if rc == 0 else f"FAILED (exit {rc})", time.time() - t0))

    print("\n=== Summary ===")
    for tag, status, secs in results:
        print(f"{tag:45s} {status:18s} {secs / 60:7.1f} min")
    if any(s.startswith("FAILED") for _, s, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
