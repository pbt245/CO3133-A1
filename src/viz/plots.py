"""Matplotlib helpers (headless backend)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def show_image(ax, img):
    """img: (C,H,W) or (H,W) array; uint8 [0,255] or float [0,1]."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr[0] if arr.shape[0] == 1 else np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        if arr.dtype == np.uint8:
            ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(arr, cmap="gray")
    else:
        ax.imshow(arr)
    ax.axis("off")


def plot_image_grid(images, titles, path, ncols=5, suptitle=None, title_colors=None, cell=2.2, fontsize=7):
    n = len(images)
    if n == 0:
        return
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell, nrows * (cell + 0.4)), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i in range(n):
        ax = axes.flat[i]
        show_image(ax, images[i])
        if titles is not None and titles[i]:
            color = title_colors[i] if title_colors is not None else "black"
            ax.set_title(titles[i], fontsize=fontsize, color=color)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    _save(fig, path)


def plot_training_curves(history, path, title=None, best_epoch=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    e = history["epoch"]
    axes[0].plot(e, history["train_loss"], label="train")
    axes[0].plot(e, history["val_loss"], label="val")
    axes[0].set_title("Cross-entropy loss")
    axes[1].plot(e, history["train_acc"], label="train (aug + dropout on)")
    axes[1].plot(e, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[2].plot(e, history["val_macro_f1"], label="val", color="tab:green")
    axes[2].set_title("Validation macro-F1")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if best_epoch is not None:
            ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"selected epoch {best_epoch}")
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    _save(fig, path)


def plot_confusion_matrix(cm, classes, path, normalize=False, title=None):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row = cm.sum(axis=1, keepdims=True)
        data = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    else:
        data = cm
    k = len(classes)
    size = max(6.0, 0.65 * k + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(k))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    thresh = data.max() / 2 if data.max() > 0 else 0.5
    for i in range(k):
        for j in range(k):
            txt = f"{data[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if data[i, j] > thresh else "black")
    if title:
        ax.set_title(title, fontsize=10)
    _save(fig, path)


def plot_class_distribution(counts: dict, classes, path, title=None):
    splits = list(counts)
    x = np.arange(len(classes))
    width = 0.8 / len(splits)
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(classes)), 4))
    for i, s in enumerate(splits):
        ax.bar(x + i * width - 0.4 + width / 2, counts[s], width, label=f"{s} (n={int(np.sum(counts[s]))})")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("images")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title or "Class distribution per split")
    _save(fig, path)


def plot_heatmap(data, row_labels, col_labels, path, title=None, fmt=".2f", cmap="viridis",
                 vmin=None, vmax=None, figsize=None):
    data = np.asarray(data, dtype=float)
    r, c = data.shape
    fig, ax = plt.subplots(figsize=figsize or (max(6, 0.7 * c + 2), max(4, 0.5 * r + 1.5)))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(c))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(r))
    ax.set_yticklabels(row_labels)
    finite = data[np.isfinite(data)]
    mid = (finite.max() + finite.min()) / 2 if finite.size else 0
    for i in range(r):
        for j in range(c):
            if np.isfinite(data[i, j]):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=7,
                        color="white" if data[i, j] < mid else "black")
    if title:
        ax.set_title(title, fontsize=10)
    _save(fig, path)
