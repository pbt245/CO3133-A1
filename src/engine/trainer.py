"""Full training procedure: optimizer, scheduler, epochs, validation, checkpointing, early stopping.

Checkpoint selection rule (identical for all models):
  keep the epoch with the best `train.checkpoint_metric` on the VALIDATION split
  (default: highest val macro-F1; the first epoch reaching the best value is kept).
Early stopping: stop after `early_stopping_patience` epochs without improvement.
The TEST split is never touched here.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from src.data.datasets import build_dataloaders
from src.engine.loops import make_grad_scaler, predict, train_one_epoch
from src.engine.metrics import classification_metrics
from src.models import build_model, count_parameters
from src.utils.env import synchronize
from src.utils.io import save_json
from src.utils.seed import set_seed
from src.viz.plots import plot_training_curves

METRIC_MODES = {"val_macro_f1": "max", "val_acc": "max", "val_loss": "min"}


def build_optimizer(model: nn.Module, optim_cfg: dict):
    """Weight decay is applied to weight matrices/kernels only (not biases, norm
    parameters, positional embeddings or the CLS token)."""
    name = str(optim_cfg.get("name", "adamw")).lower()
    lr = float(optim_cfg["lr"])
    wd = float(optim_cfg.get("weight_decay", 0.0))
    decay, no_decay = [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or pname.endswith(".bias") or "pos_encoding" in pname or "cls_token" in pname:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [g for g in ({"params": decay, "weight_decay": wd},
                          {"params": no_decay, "weight_decay": 0.0}) if g["params"]]
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=float(optim_cfg.get("momentum", 0.9)), nesterov=True)
    raise ValueError(f"Unknown optimizer '{name}'")


def build_scheduler(optimizer, sched_cfg: dict, epochs: int, steps_per_epoch: int):
    """Per-iteration LambdaLR: linear warm-up, then cosine decay to min_lr_ratio (or constant)."""
    name = str(sched_cfg.get("name", "cosine")).lower()
    total = max(1, epochs * steps_per_epoch)
    warm = int(float(sched_cfg.get("warmup_epochs", 0)) * steps_per_epoch)
    min_ratio = float(sched_cfg.get("min_lr_ratio", 0.01))

    def lr_lambda(step: int) -> float:
        if warm > 0 and step < warm:
            return (step + 1) / warm
        if name == "constant":
            return 1.0
        progress = min(1.0, (step - warm) / max(1, total - warm))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    if name not in ("cosine", "constant"):
        raise ValueError(f"Unknown scheduler '{name}'")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_training(cfg: dict, run_dir, device: torch.device, logger) -> dict:
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["seed"])
    set_seed(seed, deterministic=bool(cfg.get("deterministic", False)))
    tcfg = cfg["train"]
    show_progress = bool(tcfg.get("show_progress", True))

    loaders, _, meta = build_dataloaders(cfg, seed, device, logger)
    logger.info("Dataset %s | split %s | train/val/test = %d/%d/%d | fingerprints %s",
                meta["dataset"], meta["split_file"], meta["sizes"]["train"], meta["sizes"]["val"],
                meta["sizes"]["test"], meta["split_fingerprints"])

    model = build_model(cfg["model"], meta["in_channels"], meta["image_size"], meta["num_classes"]).to(device)
    params = count_parameters(model)
    logger.info("Model %s | trainable parameters: %s", cfg["model"]["name"], f"{params['trainable']:,}")
    (run_dir / "model_architecture.txt").write_text(
        f"{model}\n\ntrainable parameters: {params['trainable']:,}\n", encoding="utf-8")

    epochs = int(tcfg["epochs"])
    use_amp = bool(tcfg.get("amp", False)) and device.type == "cuda"
    train_criterion = nn.CrossEntropyLoss(label_smoothing=float(tcfg.get("label_smoothing", 0.0)))
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg["optim"])
    scheduler = build_scheduler(optimizer, cfg.get("scheduler", {}), epochs, len(loaders["train"]))
    scaler = make_grad_scaler(use_amp)
    grad_clip = float(tcfg.get("grad_clip") or 0.0)
    metric_name = tcfg.get("checkpoint_metric", "val_macro_f1")
    if metric_name not in METRIC_MODES:
        raise ValueError(f"checkpoint_metric must be one of {list(METRIC_MODES)}")
    mode = METRIC_MODES[metric_name]
    patience = int(tcfg.get("early_stopping_patience") or 0)
    run_id = cfg.get("run", {}).get("run_id")

    history = []
    best_value, best_epoch, no_improve = None, None, 0
    train_time_total = 0.0
    wall_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        lr_start = optimizer.param_groups[0]["lr"]
        synchronize(device)
        t0 = time.perf_counter()
        tr = train_one_epoch(model, loaders["train"], train_criterion, optimizer, scheduler, scaler,
                             device, use_amp, grad_clip, desc=f"epoch {epoch}/{epochs}",
                             show_progress=show_progress)
        synchronize(device)
        t_train = time.perf_counter() - t0
        train_time_total += t_train

        t1 = time.perf_counter()
        va = predict(model, loaders["val"], device, eval_criterion)
        t_val = time.perf_counter() - t1
        va_m = classification_metrics(va["y_true"], va["y_pred"], meta["num_classes"])

        row = {
            "epoch": epoch, "lr": lr_start,
            "train_loss": tr["loss"], "train_acc": tr["acc"],
            "val_loss": va["loss"], "val_acc": va_m["accuracy"], "val_macro_f1": va_m["macro_f1"],
            "epoch_train_time_s": t_train, "epoch_val_time_s": t_val,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        value = row[metric_name]
        improved = best_value is None or (value > best_value + 1e-6 if mode == "max" else value < best_value - 1e-6)
        state = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_metrics": {k: row[k] for k in ("val_loss", "val_acc", "val_macro_f1")},
            "checkpoint_metric": metric_name,
            "run_id": run_id,
            "model_cfg": cfg["model"],
            "meta": {k: meta[k] for k in ("dataset", "num_classes", "in_channels", "image_size")},
        }
        if improved:
            best_value, best_epoch, no_improve = value, epoch, 0
            torch.save(state, ckpt_dir / "best.pt")
        else:
            no_improve += 1
        torch.save(state, ckpt_dir / "last.pt")

        logger.info(
            "epoch %3d | lr %.2e | train loss %.4f acc %.4f | val loss %.4f acc %.4f macroF1 %.4f | %.1fs%s",
            epoch, lr_start, tr["loss"], tr["acc"], va["loss"], va_m["accuracy"], va_m["macro_f1"],
            t_train, "  *best*" if improved else "")

        if patience and no_improve >= patience:
            logger.info("Early stopping: no improvement in %s for %d epochs.", metric_name, patience)
            break

    best_row = history[best_epoch - 1]
    summary = {
        "run_id": run_id,
        "dataset": meta["dataset"],
        "experiment": cfg["experiment"]["name"],
        "model": cfg["model"]["name"],
        "seed": seed,
        "device": str(device),
        "amp": use_amp,
        "epochs_max": epochs,
        "epochs_run": len(history),
        "early_stopped": len(history) < epochs,
        "checkpoint_metric": metric_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_row["val_loss"],
        "best_val_acc": best_row["val_acc"],
        "best_val_macro_f1": best_row["val_macro_f1"],
        "train_time_s": train_time_total,
        "train_time_per_epoch_s": train_time_total / len(history),
        "wall_time_s": time.perf_counter() - wall_start,
        "parameters": params,
        "split_file": meta["split_file"],
        "split_fingerprints": meta["split_fingerprints"],
    }
    save_json(summary, run_dir / "train_summary.json")
    plot_training_curves(pd.DataFrame(history), run_dir / "curves.png",
                         title=f"{meta['dataset']} / {cfg['experiment']['name']} / seed {seed}",
                         best_epoch=best_epoch)
    logger.info("Best epoch %d | val macroF1 %.4f | val acc %.4f | train time %.1fs",
                best_epoch, best_row["val_macro_f1"], best_row["val_acc"], train_time_total)
    return summary
