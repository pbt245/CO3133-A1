#!/usr/bin/env python
"""Zip EDA, comparison outputs and run logs/metrics (NO checkpoints) to send for report writing.

Usage:  python scripts/pack_results.py --datasets fashion_mnist cifar10 [--include-predictions]
Output: outputs/results_bundle_<timestamp>.zip
"""
import argparse
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["fashion_mnist", "cifar10"])
    p.add_argument("--include-predictions", action="store_true")
    args = p.parse_args()

    bundle = OUT / f"results_bundle_{time.strftime('%Y%m%d-%H%M%S')}.zip"
    n_files = 0
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for ds in args.datasets:
            roots = [OUT / "eda" / ds, OUT / "comparison" / ds, OUT / "runs" / ds]
            for root in roots:
                if not root.exists():
                    print(f"(missing) {root.relative_to(ROOT)}")
                    continue
                for f in root.rglob("*"):
                    if f.is_dir() or "checkpoints" in f.parts:
                        continue
                    if f.name == "predictions.csv" and not args.include_predictions:
                        continue
                    zf.write(f, f.relative_to(ROOT))
                    n_files += 1
        for extra in ["requirements.txt"]:
            if (ROOT / extra).exists():
                zf.write(ROOT / extra, extra)
                n_files += 1
    print(f"{n_files} files -> {bundle} ({bundle.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
