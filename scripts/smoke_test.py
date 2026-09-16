#!/usr/bin/env python
"""Offline sanity check (no dataset download needed).

* verifies rows / cols / patches sequence conversion against manual slicing
* verifies augmentation keeps the image shape
* builds every experiment config, runs forward + backward on random input,
  prints the parameter count and token shapes
Usage:  python scripts/smoke_test.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.data.datasets import augment_image, dataset_info  # noqa: E402
from src.models import build_model, count_parameters  # noqa: E402
from src.models.sequence import image_to_sequence  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def check_sequences():
    x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).view(2, 3, 8, 8)
    rows = image_to_sequence(x, "rows")
    assert rows.shape == (2, 8, 24) and torch.equal(rows[:, 5], x[:, :, 5, :].reshape(2, -1))
    cols = image_to_sequence(x, "cols")
    assert cols.shape == (2, 8, 24) and torch.equal(cols[:, 3], x[:, :, :, 3].reshape(2, -1))
    patches = image_to_sequence(x, "patches", 4)
    assert patches.shape == (2, 4, 48)
    assert torch.equal(patches[:, 1], x[:, :, 0:4, 4:8].reshape(2, -1))   # top-right patch
    assert torch.equal(patches[:, 2], x[:, :, 4:8, 0:4].reshape(2, -1))   # bottom-left patch
    print("[ok] image_to_sequence rows / cols / patches")


def check_augment():
    x = torch.rand(1, 28, 28)
    y = augment_image(x, random_crop_padding=2, hflip=True)
    assert y.shape == x.shape
    print("[ok] augmentation keeps shape (1, 28, 28)")


def check_models():
    cfg_files = []
    for ds_dir in ("fashion_mnist", "cifar10"):
        d = ROOT / "configs" / ds_dir
        cfg_files += sorted(p for p in d.glob("*.yaml") if not p.name.startswith("_"))
        cfg_files += sorted((d / "ablations").glob("*.yaml"))
    criterion = nn.CrossEntropyLoss()
    names_by_dataset = {}
    print(f"\n{'dataset':14s} {'experiment':20s} {'group':9s} {'params':>12s}  tokens / output")
    for path in cfg_files:
        cfg = load_config(path)
        info = dataset_info(cfg["data"]["name"])
        c, s, k = info["channels"], info["image_size"], len(info["classes"])
        model = build_model(cfg["model"], c, s, k)
        x = torch.randn(4, c, s, s)
        logits = model(x)
        assert logits.shape == (4, k), logits.shape
        loss = criterion(logits, torch.randint(0, k, (4,)))
        loss.backward()
        extra = ""
        if cfg["model"]["name"] in ("lstm", "gru", "transformer"):
            seq = image_to_sequence(x, cfg["model"]["sequence_mode"], cfg["model"].get("patch_size", 4))
            extra = f"sequence {tuple(seq.shape)} -> "
        n = count_parameters(model)["trainable"]
        exp = cfg["experiment"]
        names_by_dataset.setdefault(cfg["data"]["name"], set()).add(exp["name"])
        print(f"{cfg['data']['name']:14s} {exp['name']:20s} {exp.get('group', 'main'):9s} {n:>12,d}  "
              f"{extra}logits {tuple(logits.shape)}")
        if exp.get("group") == "ablation":
            assert exp.get("reference"), f"{path} needs experiment.reference"
    for path in cfg_files:
        cfg = load_config(path)
        ref = cfg["experiment"].get("reference")
        if ref:
            assert ref in names_by_dataset[cfg["data"]["name"]], f"reference '{ref}' not found for {path}"
    print("\n[ok] all configs build, forward and backward")


if __name__ == "__main__":
    torch.manual_seed(0)
    check_sequences()
    check_augment()
    check_models()
