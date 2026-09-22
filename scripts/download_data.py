#!/usr/bin/env python
"""Download MNIST (debug), Fashion-MNIST (main) and CIFAR-10 (extension) via torchvision.

Usage:  python scripts/download_data.py --datasets mnist fashion_mnist cifar10
Files land in data/raw/ (MNIST/, FashionMNIST/, cifar-10-batches-py/).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.data.datasets import DATASET_INFO, dataset_info, load_raw
from src.utils.io import resolve_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["mnist", "fashion_mnist", "cifar10"],
                   choices=list(DATASET_INFO))
    p.add_argument("--raw-root", default="data/raw")
    args = p.parse_args()

    root = resolve_path(args.raw_root)
    root.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        info = dataset_info(name)
        for train, expected in zip((True, False), info["official_sizes"]):
            images, labels = load_raw(name, root, train=train, download=True)
            split = "train" if train else "test"
            status = "OK" if len(labels) == expected else f"UNEXPECTED (expected {expected})"
            counts = np.bincount(labels.numpy(), minlength=len(info["classes"])).tolist()
            print(f"{name:14s} {split:5s} images {tuple(images.shape)} dtype {images.dtype} "
                  f"-> {status}; per-class counts {counts}")
    print(f"\nData stored under {root}")


if __name__ == "__main__":
    main()
