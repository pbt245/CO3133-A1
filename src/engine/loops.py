"""Train / predict loops shared by every model."""
from __future__ import annotations

import contextlib
import math

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


def autocast_ctx(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device,
                    use_amp: bool = False, grad_clip: float = 0.0, desc: str = "train",
                    show_progress: bool = True) -> dict:
    """One pass over the training loader. Scheduler is stepped per iteration.
    Loss is accumulated weighted by batch size (correct mean over samples).
    Note: train accuracy is measured with dropout + augmentation active."""
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in tqdm(loader, desc=desc, leave=False, disable=not show_progress):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite training loss ({loss_value}); lower the learning rate.")
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        bs = y.size(0)
        loss_sum += loss_value * bs
        correct += (logits.detach().argmax(dim=1) == y).sum().item()
        n += bs
    return {"loss": loss_sum / max(n, 1), "acc": correct / max(n, 1), "n": n}


@torch.no_grad()
def predict(model, loader, device, criterion=None, desc: str = "eval", show_progress: bool = False) -> dict:
    """FP32 inference over a non-shuffled loader. Returns loss, labels, predictions, probabilities."""
    model.eval()
    all_logits, all_y = [], []
    loss_sum, n = 0.0, 0
    for x, y in tqdm(loader, desc=desc, leave=False, disable=not show_progress):
        x = x.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)
        logits = model(x).float()
        if criterion is not None:
            loss_sum += criterion(logits, y_dev).item() * y.size(0)
        all_logits.append(logits.cpu())
        all_y.append(y.cpu())
        n += y.size(0)
    logits = torch.cat(all_logits)
    probs = torch.softmax(logits, dim=1).numpy()
    y_true = torch.cat(all_y).numpy()
    return {
        "loss": (loss_sum / n) if criterion is not None else None,
        "y_true": y_true,
        "y_pred": probs.argmax(axis=1).astype(np.int64),
        "probs": probs,
    }
