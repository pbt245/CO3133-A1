#!/usr/bin/env python
"""Figures that support the method explanations in the report (Sec. 11.1):
  linear       : weight template per class
  mlp          : first-layer weights of 16 hidden units
  cnn          : feature maps after each conv block
  lstm / gru   : test accuracy after reading t timesteps + P(true class) over timesteps
  transformer  : CLS attention over tokens, every layer (heads averaged)

Usage:  python scripts/visualize_internals.py --run-dir outputs/runs/fashion_mnist/transformer/seed42
Output: <run-dir>/internals/
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.engine.evaluate import load_run
from src.models.sequence import patch_grid
from src.utils.env import resolve_device
from src.utils.io import resolve_path
from src.viz.plots import show_image


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def weight_grid(weights, titles, path, suptitle, ncols=5):
    """weights: (N, C, H, W) numpy."""
    n = len(weights)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2.2), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i in range(n):
        w = weights[i]
        ax = axes.flat[i]
        if w.shape[0] == 1:
            a = np.abs(w).max() + 1e-8
            ax.imshow(w[0], cmap="seismic", vmin=-a, vmax=a)
        else:
            w = np.transpose(w, (1, 2, 0))
            ax.imshow((w - w.min()) / (w.max() - w.min() + 1e-8))
        ax.set_title(titles[i], fontsize=8)
    fig.suptitle(suptitle, fontsize=10)
    save(fig, path)


def viz_linear(model, meta, out):
    c, s = meta["in_channels"], meta["image_size"]
    w = model.fc.weight.detach().cpu().view(meta["num_classes"], c, s, s).numpy()
    weight_grid(w, meta["classes"], out / "linear_class_templates.png",
                "Linear classifier: weight template per class (red = positive, blue = negative)")


def viz_mlp(model, meta, out):
    first = next(m for m in model.net if isinstance(m, nn.Linear))
    c, s = meta["in_channels"], meta["image_size"]
    w = first.weight.detach().cpu()[:16].view(-1, c, s, s).numpy()
    weight_grid(w, [f"unit {i}" for i in range(len(w))], out / "mlp_first_layer_units.png",
                "MLP: input weights of the first 16 hidden units", ncols=8)


@torch.no_grad()
def viz_cnn(model, x, raw, y, preds, classes, out, n_show=3, n_channels=8):
    maps = model.feature_maps(x)
    for i in range(min(n_show, len(x))):
        nb = len(maps)
        fig, axes = plt.subplots(nb, n_channels + 1, figsize=((n_channels + 1) * 1.5, nb * 1.7), squeeze=False)
        for b, fm in enumerate(maps):
            fmi = fm[i].cpu()
            show_image(axes[b][0], raw[i])
            axes[b][0].set_title(f"block {b + 1}: {tuple(fmi.shape)}", fontsize=7)
            idx = fmi.mean(dim=(1, 2)).topk(min(n_channels, fmi.shape[0])).indices
            for j in range(n_channels):
                ax = axes[b][j + 1]
                ax.axis("off")
                if j < len(idx):
                    ax.imshow(fmi[idx[j]].numpy(), cmap="viridis")
                    ax.set_title(f"ch {int(idx[j])}", fontsize=7)
        fig.suptitle(f"CNN feature maps (8 most active channels per block) | true {classes[y[i]]}, "
                     f"pred {classes[preds[i]]}", fontsize=9)
        save(fig, out / f"cnn_feature_maps_img{i}.png")


@torch.no_grad()
def viz_rnn(model, x, raw, y, test_ds, device, classes, out):
    if model.bidirectional:
        print("bidirectional RNN: per-step visualization skipped")
        return
    loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    correct, total = None, 0
    for xb, yb in loader:
        pred = model.forward_all_steps(xb.to(device)).argmax(-1).cpu()        # (B, T)
        c = (pred == yb.unsqueeze(1)).sum(0)
        correct = c if correct is None else correct + c
        total += len(yb)
    acc = (correct.float() / total).numpy()
    steps = np.arange(1, len(acc) + 1)
    pd.DataFrame({"timesteps_read": steps, "test_accuracy": acc}).to_csv(out / "rnn_step_accuracy.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, 100 * acc, marker=".")
    ax.set_xlabel(f"timesteps read ({model.sequence_mode})")
    ax.set_ylabel("test accuracy (%) using h_t")
    ax.set_title(f"{model.cell.upper()}: accuracy when classifying from the hidden state at step t")
    ax.grid(alpha=0.3)
    save(fig, out / "rnn_step_accuracy.png")

    logits = model.forward_all_steps(x)                                     # (n, T, K)
    probs = torch.softmax(logits.float(), -1).cpu()
    n, t, _ = probs.shape
    yt = torch.as_tensor(y).view(n, 1, 1).expand(n, t, 1)
    p_true = probs.gather(2, yt).squeeze(2).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 2]})
    for i in range(n):
        axes[0].plot(np.arange(1, t + 1), p_true[i], label=f"img {i}: {classes[y[i]]}")
    axes[0].set_xlabel("timestep")
    axes[0].set_ylabel("P(true class | steps 1..t)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].axis("off")
    k = int(np.ceil(np.sqrt(n)))
    for i in range(n):
        sub = axes[1].inset_axes([(i % k) / k, 1 - (i // k + 1) / k, 1 / k, 1 / k])
        show_image(sub, raw[i])
        sub.set_title(f"img {i}", fontsize=7)
    fig.suptitle(f"{model.cell.upper()}: how the prediction evolves while reading the image")
    save(fig, out / "rnn_prob_true_class.png")


@torch.no_grad()
def viz_transformer(model, x, raw, y, preds, classes, meta, out):
    model.set_store_attention(True)
    model(x)
    atts = [blk.attn.last_attention.float().cpu() for blk in model.blocks]    # (n, heads, N, N)
    model.set_store_attention(False)
    s = meta["image_size"]
    gh, gw = patch_grid(model.sequence_mode, s, s, model.patch_size)
    has_cls = model.cls_token is not None
    n, depth = len(x), len(atts)
    fig, axes = plt.subplots(n, depth + 1, figsize=((depth + 1) * 2, n * 2.1), squeeze=False)
    for i in range(n):
        show_image(axes[i][0], raw[i])
        axes[i][0].set_title(f"T:{classes[y[i]]}\nP:{classes[preds[i]]}", fontsize=7)
        img = raw[i]
        disp = img[0] if img.shape[0] == 1 else np.transpose(img, (1, 2, 0))
        for layer, a in enumerate(atts):
            a = a[i].mean(0)                                                   # (N, N)
            scores = a[0, 1:] if has_cls else a.mean(0)
            grid = scores.numpy().reshape(gh, gw)
            up = np.kron(grid, np.ones((s // gh, s // gw)))
            ax = axes[i][layer + 1]
            ax.imshow(disp, cmap="gray" if disp.ndim == 2 else None)
            ax.imshow(up, cmap="jet", alpha=0.5)
            ax.axis("off")
            if i == 0:
                ax.set_title(f"layer {layer + 1}", fontsize=8)
    what = "CLS → token attention" if has_cls else "mean attention received per token"
    fig.suptitle(f"Transformer ({model.sequence_mode}): {what}, heads averaged", fontsize=10)
    save(fig, out / "transformer_attention.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--n-images", type=int, default=6)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    run_dir = resolve_path(args.run_dir)
    cfg, model, _, _, datasets, meta = load_run(run_dir, device)
    out = run_dir / "internals"
    out.mkdir(parents=True, exist_ok=True)
    test = datasets["test"]
    classes = meta["classes"]
    n = min(args.n_images, len(test))
    x = torch.stack([test[i][0] for i in range(n)]).to(device)
    y = test.labels[:n].numpy()
    raw = test.images[:n].numpy()
    with torch.no_grad():
        preds = model(x).argmax(1).cpu().numpy()

    name = cfg["model"]["name"]
    if name == "linear":
        viz_linear(model, meta, out)
    elif name == "mlp":
        viz_mlp(model, meta, out)
    elif name == "cnn":
        viz_cnn(model, x, raw, y, preds, classes, out)
    elif name in ("lstm", "gru"):
        viz_rnn(model, x, raw, y, test, device, classes, out)
    elif name == "transformer":
        viz_transformer(model, x, raw, y, preds, classes, meta, out)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
