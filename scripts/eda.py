#!/usr/bin/env python
"""Exploratory data analysis required by the handbook (Sec. 13 + Sec. 3.1).

Outputs (outputs/eda/<dataset>/):
  class_distribution.csv/.png, imbalance + stratification check, input size info,
  representative_samples.png, class_mean_images.png, class_mean_correlation.png/.csv,
  pixel_histogram.png, class_brightness_stats.csv, duplicate / leakage check,
  preprocessed_train_batch.png, augmentation_examples.png, eda_summary.json/.md

Usage:  python scripts/eda.py --config configs/fashion_mnist/_base.yaml
"""
import argparse
import copy
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.datasets import build_dataloaders, dataset_info, get_split, load_raw
from src.models.sequence import image_to_sequence
from src.utils.config import apply_overrides, load_config
from src.utils.io import resolve_path, save_json
from src.utils.seed import set_seed
from src.viz.plots import plot_class_distribution, plot_heatmap, plot_image_grid

SPLITS = ("train", "val", "test")


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def image_hashes(x):
    return [hashlib.md5(x[i].numpy().tobytes()).hexdigest() for i in range(len(x))]


def duplicate_stats(hashes, labels):
    groups = defaultdict(list)
    for i, h in enumerate(hashes):
        groups[h].append(i)
    dup = [g for g in groups.values() if len(g) > 1]
    conflicting = sum(1 for g in dup if len(set(labels[g].tolist())) > 1)
    return {
        "n_images": len(hashes),
        "n_unique": len(groups),
        "duplicate_groups": len(dup),
        "images_in_duplicate_groups": int(sum(len(g) for g in dup)),
        "label_conflicting_groups": int(conflicting),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--out", default="outputs/eda")
    p.add_argument("--samples-per-class", type=int, default=10)
    args = p.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    dcfg = cfg["data"]
    name = dcfg["name"]
    info = dataset_info(name)
    classes = info["classes"]
    k = len(classes)
    out = resolve_path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)

    split, split_path = get_split(dcfg)
    tv_x, tv_y = load_raw(name, dcfg["raw_root"], train=True)
    te_x, te_y = load_raw(name, dcfg["raw_root"], train=False)
    tr_idx = torch.as_tensor(split["train_indices"], dtype=torch.long)
    va_idx = torch.as_tensor(split["val_indices"], dtype=torch.long)
    X = {"train": tv_x[tr_idx], "val": tv_x[va_idx], "test": te_x}
    Y = {"train": tv_y[tr_idx].numpy(), "val": tv_y[va_idx].numpy(), "test": te_y.numpy()}
    summary = {"dataset": name, "split_file": str(split_path)}

    # 1) class distribution, imbalance, stratification ---------------------------------
    counts = {s: np.bincount(Y[s], minlength=k) for s in SPLITS}
    rows = []
    for j, c in enumerate(classes):
        row = {"class": c}
        for s in SPLITS:
            row[f"{s}_count"] = int(counts[s][j])
            row[f"{s}_pct"] = round(100 * counts[s][j] / counts[s].sum(), 3)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "class_distribution.csv", index=False)
    plot_class_distribution(counts, classes, out / "class_distribution.png", title=f"{name}: class distribution")

    imbalance = {}
    for s in SPLITS:
        c = counts[s].astype(float)
        pr = c / c.sum()
        ent = -(pr[pr > 0] * np.log(pr[pr > 0])).sum() / np.log(k)
        imbalance[s] = {
            "min_count": int(c.min()), "min_class": classes[int(c.argmin())],
            "max_count": int(c.max()), "max_class": classes[int(c.argmax())],
            "max_min_ratio": float(c.max() / c.min()) if c.min() > 0 else float("inf"),
            "coef_variation": float(c.std() / c.mean()),
            "normalized_entropy": float(ent),
        }
    pct = {s: 100 * counts[s] / counts[s].sum() for s in SPLITS}
    stratification = {
        "max_abs_class_pct_diff_train_vs_val": float(np.abs(pct["train"] - pct["val"]).max()),
        "max_abs_class_pct_diff_train_vs_test": float(np.abs(pct["train"] - pct["test"]).max()),
    }
    summary.update({"split_sizes": {s: int(len(Y[s])) for s in SPLITS}, "imbalance": imbalance,
                    "stratification_check": stratification})

    # 2) input size ---------------------------------------------------------------------
    c_, h_, w_ = X["train"].shape[1:]
    summary["input"] = {
        "raw_shape_per_image": [int(c_), int(h_), int(w_)],
        "raw_dtype": "uint8",
        "raw_value_range_train": [int(X["train"].min()), int(X["train"].max())],
        "flattened_dim": int(c_ * h_ * w_),
        "sequence_views": {
            "rows": [int(h_), int(c_ * w_)],
            "cols": [int(w_), int(c_ * h_)],
            "patches_4x4": [int((h_ // 4) * (w_ // 4)), int(c_ * 16)],
        },
        "split": {"split_seed": split["split_seed"], "val_fraction": split["val_fraction"],
                  "stratified": split["stratified"], "protocol": split["protocol"],
                  "fingerprints": split["fingerprints"]},
        "normalization_train_only": split["normalization"],
    }

    # 3) pixel statistics (train split) --------------------------------------------------
    hist = torch.zeros(256, dtype=torch.long)
    means, nonzero = [], []
    for i in range(0, len(X["train"]), 5000):
        xb = X["train"][i:i + 5000]
        hist += torch.bincount(xb.flatten().long(), minlength=256)
        means.append(xb.float().mean(dim=(1, 2, 3)) / 255.0)
        nonzero.append((xb > 0).float().mean(dim=(1, 2, 3)))
    means = torch.cat(means).numpy()
    nonzero = torch.cat(nonzero).numpy()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(np.arange(256), hist.numpy(), width=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("pixel value (uint8)")
    ax.set_ylabel("count (log)")
    ax.set_title(f"{name}: pixel intensity histogram (train split)")
    fig.tight_layout()
    fig.savefig(out / "pixel_histogram.png", dpi=150)
    plt.close(fig)
    brightness = pd.DataFrame({
        "class": classes,
        "mean_brightness": [float(means[Y["train"] == j].mean()) for j in range(k)],
        "std_brightness": [float(means[Y["train"] == j].std()) for j in range(k)],
        "mean_nonzero_pixel_fraction": [float(nonzero[Y["train"] == j].mean()) for j in range(k)],
    })
    brightness.to_csv(out / "class_brightness_stats.csv", index=False)
    summary["pixels"] = {"zero_pixel_fraction_train": float(hist[0].item() / hist.sum().item())}

    # 4) representative samples ------------------------------------------------------------
    rng = np.random.default_rng(int(dcfg.get("split_seed", 42)))
    spc = args.samples_per_class
    imgs, titles = [], []
    for j, c in enumerate(classes):
        idx = rng.choice(np.where(Y["train"] == j)[0], size=spc, replace=False)
        for t, i in enumerate(idx):
            imgs.append(X["train"][i].numpy())
            titles.append(c if t == 0 else "")
    plot_image_grid(imgs, titles, out / "representative_samples.png", ncols=spc, cell=1.3,
                    suptitle=f"{name}: {spc} random training samples per class (row = class)")

    # 5) class mean images + similarity ------------------------------------------------------
    mean_imgs = np.stack([X["train"][torch.from_numpy(np.where(Y["train"] == j)[0])].float().mean(0).numpy()
                          for j in range(k)])
    plot_image_grid([np.clip(m, 0, 255).astype(np.uint8) for m in mean_imgs], classes,
                    out / "class_mean_images.png", ncols=min(k, 5), suptitle=f"{name}: mean training image per class")
    corr = np.corrcoef(mean_imgs.reshape(k, -1))
    pd.DataFrame(corr, index=classes, columns=classes).to_csv(out / "class_mean_correlation.csv")
    plot_heatmap(corr, classes, classes, out / "class_mean_correlation.png",
                 title="Pearson correlation between class mean images", vmin=-1, vmax=1, cmap="coolwarm")
    pairs = sorted(((corr[a, b], classes[a], classes[b]) for a in range(k) for b in range(a + 1, k)), reverse=True)
    summary["most_similar_class_means"] = [{"class_a": a, "class_b": b, "corr": float(r)} for r, a, b in pairs[:5]]

    # 6) duplicates / leakage -------------------------------------------------------------------
    h_tv = image_hashes(tv_x)
    h_te = image_hashes(te_x)
    h_tr_split = [h_tv[i] for i in split["train_indices"]]
    h_va_split = [h_tv[i] for i in split["val_indices"]]
    set_tr = set(h_tr_split)
    summary["duplicates"] = {
        "official_train": duplicate_stats(h_tv, tv_y.numpy()),
        "official_test": duplicate_stats(h_te, te_y.numpy()),
        "val_images_identical_to_a_train_image": int(sum(h in set_tr for h in h_va_split)),
        "test_images_identical_to_a_train_image": int(sum(h in set_tr for h in h_te)),
    }

    # 7) one preprocessed batch + augmentation ---------------------------------------------------
    cfg_eda = copy.deepcopy(cfg)
    cfg_eda["data"]["num_workers"] = 0
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    loaders, datasets, meta = build_dataloaders(cfg_eda, seed, torch.device("cpu"))
    xb, yb = next(iter(loaders["train"]))
    mean_t = torch.tensor(meta["normalization"]["mean"]).view(1, -1, 1, 1)
    std_t = torch.tensor(meta["normalization"]["std"]).view(1, -1, 1, 1)
    summary["preprocessed_batch"] = {
        "x_shape": list(xb.shape), "x_dtype": str(xb.dtype).replace("torch.", ""),
        "y_shape": list(yb.shape), "y_dtype": str(yb.dtype).replace("torch.", ""),
        "x_min": float(xb.min()), "x_max": float(xb.max()),
        "x_mean": float(xb.mean()), "x_std": float(xb.std()),
        "augmentation_train": dcfg.get("augment"),
        "sequence_shapes": {
            "rows": list(image_to_sequence(xb, "rows").shape),
            "cols": list(image_to_sequence(xb, "cols").shape),
            "patches_4x4": list(image_to_sequence(xb, "patches", 4).shape),
        },
    }
    disp = ((xb[:16] * std_t + mean_t).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    plot_image_grid(list(disp), [classes[int(t)] for t in yb[:16]], out / "preprocessed_train_batch.png",
                    ncols=8, cell=1.5, suptitle="First training batch after preprocessing (de-normalized for display)")
    base_img = datasets["train"].images[0].numpy()
    aug = [base_img] + [((datasets["train"][0][0].unsqueeze(0) * std_t + mean_t).clamp(0, 1) * 255)
                        .round().to(torch.uint8)[0].numpy() for _ in range(7)]
    plot_image_grid(aug, ["original"] + [f"augmented {i}" for i in range(1, 8)], out / "augmentation_examples.png",
                    ncols=8, cell=1.5, suptitle="Same training image, 7 random augmentations")

    save_json(summary, out / "eda_summary.json")

    # markdown ---------------------------------------------------------------------------------
    md = [f"# EDA - {name}", "",
          f"Split file: `{split_path.name}` (split_seed={split['split_seed']}, val_fraction={split['val_fraction']}, stratified)", "",
          "## Split sizes and class distribution", "",
          md_table(["class"] + [f"{s} (n, %)" for s in SPLITS],
                   [[r["class"]] + [f"{r[f'{s}_count']} ({r[f'{s}_pct']:.2f}%)" for s in SPLITS] for r in rows]),
          "", "## Imbalance", "",
          md_table(["split", "min class (n)", "max class (n)", "max/min", "CV", "normalized entropy"],
                   [[s, f"{v['min_class']} ({v['min_count']})", f"{v['max_class']} ({v['max_count']})",
                     f"{v['max_min_ratio']:.3f}", f"{v['coef_variation']:.4f}", f"{v['normalized_entropy']:.4f}"]
                    for s, v in imbalance.items()]),
          "", f"Stratification check: {stratification}", "",
          "## Input", "", f"```\n{summary['input']}\n```", "",
          "## Most similar class mean images", "",
          md_table(["class A", "class B", "corr"],
                   [[d["class_a"], d["class_b"], f"{d['corr']:.3f}"] for d in summary["most_similar_class_means"]]),
          "", "## Duplicates / leakage", "", f"```\n{summary['duplicates']}\n```", "",
          "## Preprocessed batch", "", f"```\n{summary['preprocessed_batch']}\n```", ""]
    (out / "eda_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nEDA written to {out}")


if __name__ == "__main__":
    main()
