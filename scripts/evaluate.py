#!/usr/bin/env python
"""(Re-)evaluate the best checkpoint of a finished run.

Usage:  python scripts/evaluate.py --run-dir outputs/runs/fashion_mnist/cnn/seed42 [--split test|val]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.engine.evaluate import evaluate_run  # noqa: E402
from src.utils.env import resolve_device  # noqa: E402
from src.utils.io import resolve_path  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    evaluate_run(resolve_path(args.run_dir), resolve_device(args.device), split=args.split)


if __name__ == "__main__":
    main()
