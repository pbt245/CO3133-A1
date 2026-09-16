# CO3133 Assignment 1 - From Linear Models to Modern Sequence Models

Comparative study for image classification (Deep Learning and Its Applications, CO3133, Semester 261,
HCMUT). Five mandatory models - linear/softmax, MLP, CNN, LSTM/GRU, Transformer - trained under one
data split and one evaluation protocol.

- Debug dataset: **MNIST** · Main comparison: **Fashion-MNIST** · Extension dataset: **CIFAR-10**
- Chosen optional extensions: **LSTM vs GRU**, **CIFAR-10 experiments**
- Assignment page: `[link to GitHub Pages assignment1.html]` · Report: `[link]` · AI usage log: `AI_USAGE.md`

## Repository structure

```
configs/
  common.yaml                 shared defaults (split seed, training protocol, eval settings)
  fashion_mnist/_base.yaml    dataset settings; linear|mlp|cnn|lstm|gru|transformer.yaml
  fashion_mnist/ablations/    5 ablation configs (reference, changed factor, hypothesis inside)
  cifar10/                    extension configs
data/splits/                  fixed split indices + train-only normalization (commit these)
docs/EXPERIMENT_PLAN.md       hypotheses, fixed/changed factors, decision criteria
scripts/
  smoke_test.py               offline check of all models and configs (no data needed)
  download_data.py            torchvision download
  make_splits.py              stratified train/val split
  eda.py                      exploratory data analysis
  train.py                    train one config + seed, then evaluate on test
  evaluate.py                 re-evaluate a finished run
  run_experiments.py          run many configs × seeds
  compare.py                  mean ± std tables, LSTM-vs-GRU, ablations, McNemar, figures
  visualize_internals.py      class templates, feature maps, RNN step accuracy, attention maps
  pack_results.py             zip results (without checkpoints)
src/
  data/datasets.py            Dataset, split, preprocessing, augmentation, DataLoaders
  models/                     linear.py mlp.py cnn.py rnn.py transformer.py sequence.py
  engine/                     loops.py trainer.py evaluate.py metrics.py inference.py
  utils/ viz/
```

Self-implemented: all model classes, the Transformer attention/encoder/positional encodings, the
data pipeline, training/evaluation loops, metrics aggregation. From libraries: `torch.nn` layers
(Linear, Conv2d, BatchNorm, LayerNorm, Dropout, LSTM, GRU), AdamW, torchvision dataset download,
scikit-learn (stratified split, F1/confusion matrix), SciPy (McNemar test).

## 1. Installation

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate      Windows: .venv\Scripts\activate
# GPU: install the CUDA build of torch/torchvision first (https://pytorch.org/get-started/locally/)
pip install -r requirements.txt
pip freeze > requirements-lock.txt          # exact versions used for the reported results
python scripts/smoke_test.py
```

## 2. Dataset preparation

```bash
python scripts/download_data.py --datasets mnist fashion_mnist cifar10
python scripts/make_splits.py   --datasets mnist fashion_mnist cifar10 --split-seed 42 --val-fraction 0.1
python scripts/eda.py --config configs/fashion_mnist/_base.yaml
python scripts/eda.py --config configs/cifar10/_base.yaml
```

Split protocol: official training set → stratified train (90%) / validation (10%) with `split_seed=42`;
official test set → test. Split unit = image. Normalization mean/std are computed on the train split only.
Every run records the split fingerprint.

## 3. Training and evaluation

```bash
# debug on MNIST (few epochs)
python scripts/train.py --config configs/fashion_mnist/mlp.yaml --set data.name=mnist data.augment.hflip=false train.epochs=3 experiment.group=debug

# single run
python scripts/train.py --config configs/fashion_mnist/cnn.yaml --seed 42

# main comparison, 3 seeds
python scripts/run_experiments.py --dataset fashion_mnist --group main --seeds 42 123 2026
# ablations
python scripts/run_experiments.py --dataset fashion_mnist --group ablation --seeds 42 123 2026
# extension
python scripts/run_experiments.py --dataset cifar10 --group main --seeds 42 123 2026

# re-evaluate a run
python scripts/evaluate.py --run-dir outputs/runs/fashion_mnist/cnn/seed42
```

Training protocol (all models): AdamW, batch 128, cosine schedule with linear warm-up, gradient clipping 1.0,
mixed precision on CUDA, early stopping (patience 10 FMNIST / 15 CIFAR-10).
**Checkpoint selection rule: highest validation macro-F1** (`checkpoints/best.pt`). The test set is used once.

## 4. Analysis

```bash
python scripts/compare.py --dataset fashion_mnist
python scripts/compare.py --dataset cifar10
python scripts/visualize_internals.py --run-dir outputs/runs/fashion_mnist/transformer/seed42
python scripts/pack_results.py --datasets fashion_mnist cifar10
```

## 5. Traceability

Each run folder `outputs/runs/<dataset>/<experiment>/seed<seed>/` contains the resolved `config.yaml`
(with `run_id` and CLI overrides), `env.json` (hardware, CUDA/cuDNN, library versions, git commit and dirty
flag), `train.log`, `history.csv`, `train_summary.json`, and `eval/test_metrics.json` (split fingerprint,
checkpoint epoch and SHA-256). `outputs/comparison/<dataset>/runs.csv` maps every reported number to its
run_id, commit, split fingerprint and checkpoint hash.

Recommended: commit all code and `data/splits/`, tag it (`git tag a1-final`), then run the final experiments
on the clean tree.

Seeds: data split seed 42 (fixed); training seeds 42, 123, 2026. `deterministic: true` in the config enables
deterministic cuDNN/PyTorch kernels at some speed cost; otherwise GPU results may vary slightly between runs.

## 6. Hardware

`[fill in from env.json: GPU / CPU, RAM, OS, CUDA, PyTorch version]`

## 7. Checkpoints

Checkpoints are not stored in git. Download: `[link to GitHub Release / Google Drive / Hugging Face]`.
Place each `best.pt` in `outputs/runs/<dataset>/<experiment>/seed<seed>/checkpoints/` together with the
committed run files, then run `scripts/evaluate.py`. To rebuild instead, rerun the training commands above
at the tagged commit.
