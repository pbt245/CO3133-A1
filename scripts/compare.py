#!/usr/bin/env python
"""Aggregate all finished runs of one dataset into report-ready tables and figures.

Usage:  python scripts/compare.py --dataset fashion_mnist
Outputs (outputs/comparison/<dataset>/):
  runs.csv                  one row per run (traceability: run_id, commit, split fingerprint)
  summary.csv / summary.md  mean ± std over seeds (main models + ablations)
  main_curves.png           train loss / val loss / val macro-F1 per model (lowest seed)
  f1_vs_params.png, f1_vs_latency.png, f1_vs_train_time.png
  per_class_f1.csv/.png     mean per-class F1 per experiment
  mcnemar.csv               paired significance tests between main models (shared seed)
  lstm_vs_gru.md, lstm_vs_gru_curves.png   extension: LSTM vs GRU
  ablations.md              each ablation vs its reference experiment
"""
import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.engine.metrics import mcnemar_test
from src.utils.config import load_config
from src.utils.io import load_json, resolve_path
from src.viz.plots import plot_heatmap

MAIN_ORDER = ["linear", "mlp", "cnn", "lstm", "gru", "transformer"]
METRICS = ["test_accuracy", "test_macro_f1", "test_loss", "params", "best_epoch", "epochs_run",
           "train_time_s", "train_time_per_epoch_s", "throughput_img_s", "latency_ms_bs1"]


def order_key(name):
    return (MAIN_ORDER.index(name) if name in MAIN_ORDER else len(MAIN_ORDER), name)


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def pm(mean, std, scale=1.0, digits=2):
    if pd.isna(mean):
        return "n/a"
    if pd.isna(std):
        return f"{mean * scale:.{digits}f}"
    return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}"


def collect_runs(dataset_dir: Path) -> pd.DataFrame:
    rows = []
    for mpath in sorted(dataset_dir.glob("*/seed*/eval/test_metrics.json")):
        run_dir = mpath.parent.parent
        if not (run_dir / "train_summary.json").exists():
            continue
        m = load_json(mpath)
        s = load_json(run_dir / "train_summary.json")
        cfg = load_config(run_dir / "config.yaml")
        env = load_json(run_dir / "env.json") if (run_dir / "env.json").exists() else {}
        exp = cfg["experiment"]
        git = env.get("git") or {}
        gpus = env.get("gpus") or []
        try:
            rel = str(run_dir.relative_to(ROOT).as_posix())
        except ValueError:
            rel = str(run_dir)
        rows.append({
            "experiment": exp["name"], "group": exp.get("group", "main"), "reference": exp.get("reference"),
            "changed_factor": exp.get("changed_factor"), "hypothesis": exp.get("hypothesis"),
            "model": cfg["model"]["name"], "seed": int(cfg["seed"]),
            "params": int(m["parameters"]["trainable"]),
            "test_accuracy": m["accuracy"], "test_macro_f1": m["macro_f1"], "test_loss": m["loss"],
            "acc_ci_low": m["bootstrap_ci"]["accuracy"][0], "acc_ci_high": m["bootstrap_ci"]["accuracy"][1],
            "f1_ci_low": m["bootstrap_ci"]["macro_f1"][0], "f1_ci_high": m["bootstrap_ci"]["macro_f1"][1],
            "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"],
            "best_val_macro_f1": s["best_val_macro_f1"],
            "train_time_s": s["train_time_s"], "train_time_per_epoch_s": s["train_time_per_epoch_s"],
            "throughput_img_s": m["inference"]["throughput_img_per_s"],
            "latency_ms_bs1": m["inference"]["latency_ms_bs1_mean"],
            "device": s.get("device"), "gpu": gpus[0]["name"] if gpus else None, "amp": s.get("amp"),
            "run_id": cfg.get("run", {}).get("run_id"), "commit": git.get("commit"), "dirty": git.get("dirty"),
            "split_fingerprint_train": (m.get("split_fingerprints") or {}).get("train"),
            "checkpoint_sha256": m["checkpoint"]["sha256"],
            "run_dir": rel, "_run_dir": run_dir,
        })
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("experiment")
    agg = g[METRICS].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = g.size()
    agg["seeds"] = g["seed"].apply(lambda s: " ".join(map(str, sorted(s))))
    meta = g[["group", "model", "reference", "changed_factor", "hypothesis"]].first()
    out = meta.join(agg)
    order = sorted(out.index, key=order_key)
    return out.loc[order].reset_index()


def run_dir_of(df, exp, seed):
    return df[(df.experiment == exp) & (df.seed == seed)]["_run_dir"].iloc[0]


def summary_rows(agg):
    rows = []
    for _, r in agg.iterrows():
        rows.append([
            r["experiment"], r["seeds"], f"{int(r['params_mean']):,}",
            pm(r["test_accuracy_mean"], r["test_accuracy_std"], 100),
            pm(r["test_macro_f1_mean"], r["test_macro_f1_std"], 100),
            pm(r["best_epoch_mean"], r["best_epoch_std"], 1, 1),
            pm(r["train_time_per_epoch_s_mean"], r["train_time_per_epoch_s_std"], 1, 1),
            pm(r["train_time_s_mean"], r["train_time_s_std"], 1, 0),
            pm(r["throughput_img_s_mean"], r["throughput_img_s_std"], 1, 0),
            pm(r["latency_ms_bs1_mean"], r["latency_ms_bs1_std"], 1, 2),
        ])
    return rows


SUMMARY_HEADERS = ["Experiment", "Seeds", "Params", "Test acc (%)", "Test macro-F1 (%)", "Best epoch",
                   "Train s/epoch", "Train time (s)", "Throughput (img/s)", "Latency bs=1 (ms)"]


def plot_curves(df, experiments, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for exp in experiments:
        seed = sorted(df[df.experiment == exp].seed)[0]
        h = pd.read_csv(run_dir_of(df, exp, seed) / "history.csv")
        axes[0].plot(h.epoch, h.train_loss, label=f"{exp} (s{seed})")
        axes[1].plot(h.epoch, h.val_loss, label=f"{exp} (s{seed})")
        axes[2].plot(h.epoch, h.val_macro_f1, label=f"{exp} (s{seed})")
    for ax, t in zip(axes, ["Train loss", "Validation loss", "Validation macro-F1"]):
        ax.set_title(t)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scatter(agg, xcol, path, xlabel, title, logx=True):
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in agg.iterrows():
        marker = "o" if r["group"] == "main" else "x"
        ax.errorbar(r[f"{xcol}_mean"], 100 * r["test_macro_f1_mean"],
                    yerr=0 if pd.isna(r["test_macro_f1_std"]) else 100 * r["test_macro_f1_std"],
                    fmt=marker, capsize=3)
        ax.annotate(r["experiment"], (r[f"{xcol}_mean"], 100 * r["test_macro_f1_mean"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test macro-F1 (%)  mean ± std over seeds")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_class_f1(df, experiments):
    table = {}
    classes = None
    for exp in experiments:
        frames = []
        for rd in df[df.experiment == exp]["_run_dir"]:
            rep = pd.read_csv(rd / "eval" / "classification_report.csv")
            classes = rep["class"].tolist()
            frames.append(rep["f1"].values)
        table[exp] = np.mean(frames, axis=0)
    return pd.DataFrame(table, index=classes).T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--runs-root", default="outputs/runs")
    p.add_argument("--out", default="outputs/comparison")
    args = p.parse_args()

    dataset_dir = resolve_path(args.runs_root) / args.dataset
    out = resolve_path(args.out) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    df = collect_runs(dataset_dir)
    if df.empty:
        print(f"No finished runs under {dataset_dir}")
        return
    df.drop(columns=["_run_dir"]).to_csv(out / "runs.csv", index=False)
    agg = aggregate(df)
    agg.to_csv(out / "summary.csv", index=False)
    main_exps = [e for e in agg[agg.group == "main"]["experiment"]]
    abl = agg[agg.group == "ablation"]

    md = [f"# Comparison - {args.dataset}", "",
          "Mean ± sample std over training seeds (same data split for every run). "
          "Checkpoint = best validation macro-F1. Timing in FP32 on the device listed in runs.csv.", "",
          "## Main models", "", md_table(SUMMARY_HEADERS, summary_rows(agg[agg.group == "main"]))]
    if not abl.empty:
        md += ["", "## Ablations", "", md_table(SUMMARY_HEADERS, summary_rows(abl))]
    dirty = df[df.dirty == True]
    md += ["", "## Traceability", "",
           f"Commits used: {sorted(set(c for c in df.commit if c))}",
           f"Runs with uncommitted changes: {len(dirty)}",
           f"Train split fingerprints: {sorted(set(f for f in df.split_fingerprint_train if f))}"]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    if main_exps:
        plot_curves(df, main_exps, out / "main_curves.png", f"{args.dataset}: main models")
        scatter(agg[agg.group == "main"], "params", out / "f1_vs_params.png", "trainable parameters (log)",
                "Macro-F1 vs model size")
        scatter(agg[agg.group == "main"], "latency_ms_bs1", out / "f1_vs_latency.png",
                "latency per image, batch size 1 (ms, log)", "Macro-F1 vs inference latency")
        scatter(agg[agg.group == "main"], "train_time_s", out / "f1_vs_train_time.png",
                "total training time (s, log)", "Macro-F1 vs training cost")

    all_exps = list(agg["experiment"])
    pcf = per_class_f1(df, all_exps)
    pcf.to_csv(out / "per_class_f1.csv")
    plot_heatmap(100 * pcf.values, list(pcf.index), list(pcf.columns), out / "per_class_f1.png",
                 title="Per-class test F1 (%) - mean over seeds", fmt=".1f", vmin=0, vmax=100)

    # McNemar tests between main models
    mc_rows = []
    for a, b in itertools.combinations(main_exps, 2):
        common = sorted(set(df[df.experiment == a].seed) & set(df[df.experiment == b].seed))
        if not common:
            continue
        s = common[0]
        pa = pd.read_csv(run_dir_of(df, a, s) / "eval" / "predictions.csv")
        pb = pd.read_csv(run_dir_of(df, b, s) / "eval" / "predictions.csv")
        if not np.array_equal(pa.y_true.values, pb.y_true.values):
            print(f"warning: test labels differ between {a} and {b}; skipping McNemar")
            continue
        r = mcnemar_test(pa.y_true.values, pa.y_pred.values, pb.y_pred.values)
        mc_rows.append({"model_a": a, "model_b": b, "seed": s,
                        "acc_a": float((pa.y_true == pa.y_pred).mean()),
                        "acc_b": float((pb.y_true == pb.y_pred).mean()),
                        "a_right_b_wrong": r["b"], "a_wrong_b_right": r["c"],
                        "statistic": r["statistic"], "p_value": r["p_value"], "method": r["method"]})
    mc = pd.DataFrame(mc_rows)
    if not mc.empty:
        mc.to_csv(out / "mcnemar.csv", index=False)

    # Extension: LSTM vs GRU
    if {"lstm", "gru"} <= set(agg.experiment):
        L = agg[agg.experiment == "lstm"].iloc[0]
        G = agg[agg.experiment == "gru"].iloc[0]
        items = [("Trainable parameters", "params", 1, 0), ("Test accuracy (%)", "test_accuracy", 100, 2),
                 ("Test macro-F1 (%)", "test_macro_f1", 100, 2), ("Test loss", "test_loss", 1, 4),
                 ("Best epoch", "best_epoch", 1, 1), ("Train time per epoch (s)", "train_time_per_epoch_s", 1, 2),
                 ("Total train time (s)", "train_time_s", 1, 1), ("Throughput (img/s)", "throughput_img_s", 1, 0),
                 ("Latency bs=1 (ms)", "latency_ms_bs1", 1, 3)]
        rows = []
        for label, col, sc, dg in items:
            lm, gm = L[f"{col}_mean"], G[f"{col}_mean"]
            rel = f"{100 * (gm - lm) / lm:+.1f}%" if lm else "n/a"
            rows.append([label, pm(lm, L[f"{col}_std"], sc, dg), pm(gm, G[f"{col}_std"], sc, dg),
                         f"{(gm - lm) * sc:+.{dg}f} ({rel})"])
        lines = [f"# LSTM vs GRU - {args.dataset}", "",
                 f"Seeds LSTM: {L['seeds']} | GRU: {G['seeds']}. Only the recurrent cell differs.", "",
                 md_table(["Metric", "LSTM", "GRU", "GRU − LSTM"], rows), "",
                 "## Per-class F1 (%)", "",
                 md_table(["Class", "LSTM", "GRU", "GRU − LSTM"],
                          [[c, f"{100 * pcf.loc['lstm', c]:.2f}", f"{100 * pcf.loc['gru', c]:.2f}",
                            f"{100 * (pcf.loc['gru', c] - pcf.loc['lstm', c]):+.2f}"] for c in pcf.columns])]
        if not mc.empty:
            sel = mc[((mc.model_a == "lstm") & (mc.model_b == "gru")) | ((mc.model_a == "gru") & (mc.model_b == "lstm"))]
            if not sel.empty:
                r = sel.iloc[0]
                lines += ["", f"McNemar ({r['method']}, seed {r['seed']}): {r['model_a']} right & {r['model_b']} wrong = "
                              f"{r['a_right_b_wrong']}, opposite = {r['a_wrong_b_right']}, p = {r['p_value']:.4g}"]
        (out / "lstm_vs_gru.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        plot_curves(df, ["lstm", "gru"], out / "lstm_vs_gru_curves.png", f"{args.dataset}: LSTM vs GRU")

    # Ablations vs reference
    if not abl.empty:
        rows = []
        for _, r in abl.iterrows():
            ref = agg[agg.experiment == r["reference"]]
            if ref.empty:
                continue
            ref = ref.iloc[0]
            common = sorted(set(df[df.experiment == r["experiment"]].seed) & set(df[df.experiment == ref["experiment"]].seed))
            rows.append([
                r["experiment"], ref["experiment"], r["changed_factor"],
                pm(ref["test_macro_f1_mean"], ref["test_macro_f1_std"], 100),
                pm(r["test_macro_f1_mean"], r["test_macro_f1_std"], 100),
                f"{100 * (r['test_macro_f1_mean'] - ref['test_macro_f1_mean']):+.2f}",
                f"{100 * (r['test_accuracy_mean'] - ref['test_accuracy_mean']):+.2f}",
                f"{int(ref['params_mean']):,} → {int(r['params_mean']):,}",
                " ".join(map(str, common)),
            ])
        lines = [f"# Ablations - {args.dataset}", "",
                 md_table(["Ablation", "Reference", "Changed factor", "Ref macro-F1 (%)", "Abl macro-F1 (%)",
                           "Δ macro-F1 (pp)", "Δ acc (pp)", "Params", "Common seeds"], rows), "",
                 "## Pre-registered hypotheses", ""]
        lines += [f"- **{r['experiment']}**: {r['hypothesis']}" for _, r in abl.iterrows()]
        (out / "ablations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print((out / "summary.md").read_text(encoding="utf-8"))
    print(f"\nComparison written to {out}")


if __name__ == "__main__":
    main()
