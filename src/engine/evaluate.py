"""Evaluate the selected checkpoint of a run on the test split and write every artifact
needed for the report: metrics, bootstrap CI, per-class report, confusion matrices,
predictions, top confusions, correct / hard-correct / incorrect example grids, inference cost."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.datasets import build_datasets
from src.engine.inference import measure_inference
from src.engine.loops import predict
from src.engine.metrics import bootstrap_ci, classification_metrics, top_confusions
from src.models import build_model, count_parameters
from src.utils.config import load_config
from src.utils.env import git_info, synchronize
from src.utils.io import save_json, sha256_file
from src.utils.seed import set_seed
from src.viz.plots import plot_confusion_matrix, plot_image_grid


def load_run(run_dir, device, checkpoint: str = "best.pt"):
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    datasets, meta = build_datasets(cfg)
    model = build_model(cfg["model"], meta["in_channels"], meta["image_size"], meta["num_classes"])
    ckpt_path = run_dir / "checkpoints" / checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return cfg, model, ckpt, ckpt_path, datasets, meta


def evaluate_run(run_dir, device, split: str = "test", logger=None) -> dict:
    log = logger.info if logger is not None else print
    run_dir = Path(run_dir)
    cfg, model, ckpt, ckpt_path, datasets, meta = load_run(run_dir, device)
    seed = int(cfg["seed"])
    set_seed(seed)
    ecfg = cfg.get("eval", {})
    classes = meta["classes"]
    k = meta["num_classes"]
    dataset = datasets[split]
    out_dir = run_dir / ("eval" if split == "test" else f"eval_{split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(dataset, batch_size=int(cfg["train"].get("eval_batch_size", 512)), shuffle=False,
                        num_workers=int(cfg["data"].get("num_workers", 0)), pin_memory=device.type == "cuda")
    synchronize(device)
    t0 = time.perf_counter()
    res = predict(model, loader, device, nn.CrossEntropyLoss())
    synchronize(device)
    eval_pass_time = time.perf_counter() - t0

    y_true, y_pred, probs = res["y_true"], res["y_pred"], res["probs"]
    m = classification_metrics(y_true, y_pred, k)
    ci = bootstrap_ci(y_true, y_pred, k, n_boot=int(ecfg.get("bootstrap_samples", 1000)), seed=seed)
    cm = m["confusion_matrix"]

    # ---- per-class report and confusion matrices
    report = pd.DataFrame({
        "class": classes,
        "precision": m["per_class"]["precision"],
        "recall": m["per_class"]["recall"],
        "f1": m["per_class"]["f1"],
        "support": m["per_class"]["support"],
    })
    report.to_csv(out_dir / "classification_report.csv", index=False)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out_dir / "confusion_matrix.csv")
    title = f"{meta['dataset']} / {cfg['experiment']['name']} / seed {seed}"
    plot_confusion_matrix(cm, classes, out_dir / "confusion_matrix.png", normalize=False, title=title)
    plot_confusion_matrix(cm, classes, out_dir / "confusion_matrix_normalized.png", normalize=True,
                          title=title + " (row-normalized)")

    # ---- predictions
    conf = probs.max(axis=1)
    prob_true = probs[np.arange(len(y_true)), y_true]
    correct = y_true == y_pred
    pd.DataFrame({
        "index": np.arange(len(y_true)), "y_true": y_true, "y_pred": y_pred,
        "true_name": [classes[i] for i in y_true], "pred_name": [classes[i] for i in y_pred],
        "confidence": conf, "prob_true_class": prob_true, "correct": correct,
    }).to_csv(out_dir / "predictions.csv", index=False)

    # ---- error analysis
    confusions = top_confusions(cm, classes, top_k=15)
    confusions.to_csv(out_dir / "top_confusions.csv", index=False)
    errors = ~correct
    error_summary = {
        "n_errors": int(errors.sum()),
        "error_rate": float(errors.mean()),
        "mean_confidence_correct": float(conf[correct].mean()) if correct.any() else None,
        "mean_confidence_errors": float(conf[errors].mean()) if errors.any() else None,
        "high_confidence_errors_conf_gt_0.9": int((errors & (conf > 0.9)).sum()),
        "top_confusions": confusions.head(5).to_dict("records"),
    }

    # ---- example grids (raw, un-normalized images)
    images = dataset.images
    n_ex = int(ecfg.get("n_examples", 25))
    rng = np.random.default_rng(seed)
    correct_idx = np.where(correct)[0]
    wrong_idx = np.where(errors)[0]

    def grid(indices, filename, heading):
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) == 0:
            return
        titles = [f"T: {classes[y_true[i]]}\nP: {classes[y_pred[i]]} ({conf[i]:.2f})" for i in indices]
        colors = ["tab:green" if correct[i] else "tab:red" for i in indices]
        plot_image_grid([images[i].numpy() for i in indices], titles, out_dir / filename,
                        ncols=5, suptitle=f"{title}: {heading}", title_colors=colors)

    if len(correct_idx):
        grid(np.sort(rng.choice(correct_idx, size=min(n_ex, len(correct_idx)), replace=False)),
             "examples_correct_random.png", "random correct predictions")
        grid(correct_idx[np.argsort(conf[correct_idx])[:n_ex]],
             "examples_correct_low_confidence.png", "difficult but correct (lowest confidence)")
    if len(wrong_idx):
        grid(np.sort(rng.choice(wrong_idx, size=min(n_ex, len(wrong_idx)), replace=False)),
             "examples_incorrect_random.png", "random incorrect predictions")
        grid(wrong_idx[np.argsort(-conf[wrong_idx])[:n_ex]],
             "examples_incorrect_high_confidence.png", "most confident errors")

    # ---- inference cost
    icfg = ecfg.get("inference", {})
    inference = measure_inference(
        model, dataset, device,
        batch_size=int(icfg.get("batch_size", 256)), warmup_iters=int(icfg.get("warmup_iters", 10)),
        timed_iters=int(icfg.get("timed_iters", 50)), latency_samples=int(icfg.get("latency_samples", 200)),
        amp=bool(icfg.get("amp", False)),
    )

    metrics = {
        "run_id": cfg.get("run", {}).get("run_id"),
        "dataset": meta["dataset"],
        "experiment": cfg["experiment"]["name"],
        "model": cfg["model"]["name"],
        "seed": seed,
        "split": split,
        "n_samples": int(len(y_true)),
        "loss": res["loss"],
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "weighted_f1": m["weighted_f1"],
        "bootstrap_ci": ci,
        "per_class": report.to_dict("records"),
        "error_summary": error_summary,
        "parameters": count_parameters(model),
        "inference": inference,
        "eval_pass_time_s": eval_pass_time,
        "checkpoint": {
            "file": str(ckpt_path.relative_to(run_dir)),
            "epoch": ckpt.get("epoch"),
            "selection_metric": ckpt.get("checkpoint_metric"),
            "val_metrics": ckpt.get("val_metrics"),
            "sha256": sha256_file(ckpt_path),
        },
        "split_file": meta["split_file"],
        "split_fingerprints": meta["split_fingerprints"],
        "git_at_eval": git_info(),
    }
    name = "test_metrics.json" if split == "test" else f"{split}_metrics.json"
    save_json(metrics, out_dir / name)
    log(f"[{split}] acc {m['accuracy']:.4f} (95% CI {ci['accuracy'][0]:.4f}-{ci['accuracy'][1]:.4f}) | "
        f"macroF1 {m['macro_f1']:.4f} (95% CI {ci['macro_f1'][0]:.4f}-{ci['macro_f1'][1]:.4f}) | "
        f"throughput {inference['throughput_img_per_s']:.0f} img/s | "
        f"latency bs1 {inference['latency_ms_bs1_mean']:.2f} ms")
    return metrics
