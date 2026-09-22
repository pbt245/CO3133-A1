#!/usr/bin/env python
"""Create the fixed, stratified train/val split (+ train-only normalization stats).

Usage:  python scripts/make_splits.py --datasets mnist fashion_mnist cifar10
Commit the resulting data/splits/*.json so every run is traceable to the same split.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import DATASET_INFO, create_split, split_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["mnist", "fashion_mnist", "cifar10"],
                   choices=list(DATASET_INFO))
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--split-dir", default="data/splits")
    p.add_argument("--force", action="store_true", help="overwrite an existing split file")
    args = p.parse_args()

    for name in args.datasets:
        path = split_file(args.split_dir, name, args.split_seed, args.val_fraction)
        if path.exists() and not args.force:
            print(f"{name}: split already exists at {path} (use --force to recreate)")
            continue
        payload, path = create_split(name, args.raw_root, args.split_dir, args.split_seed, args.val_fraction)
        print(f"{name}: train/val/test = {payload['n_train']}/{payload['n_val']}/{payload['n_test']} | "
              f"mean {[round(v, 4) for v in payload['normalization']['mean']]} "
              f"std {[round(v, 4) for v in payload['normalization']['std']]} | "
              f"fingerprints {payload['fingerprints']} -> {path}")


if __name__ == "__main__":
    main()
