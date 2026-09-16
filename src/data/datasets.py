"""Dataset loading, fixed train/val/test split, preprocessing, augmentation, DataLoaders.

Split protocol (identical for every model of a dataset):
  * official training set -> stratified train / val using `split_seed` and `val_fraction`
  * official test set     -> test (only used once, after checkpoint selection)
  * split indices + normalization statistics are saved to data/splits/*.json
  * mean/std are computed on the TRAIN split only (no leakage from val/test)

Preprocessing (per sample):
  uint8 [0,255] -> float32 [0,1] -> (train only) augmentation -> per-channel standardization
Output tensor shape: (C, H, W)  i.e. (1, 28, 28) for (Fashion-)MNIST, (3, 32, 32) for CIFAR-10.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tv_datasets

from src.utils.io import load_json, resolve_path, save_json
from src.utils.seed import seed_worker

FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

DATASET_INFO = {
    "mnist": {"cls": tv_datasets.MNIST, "channels": 1, "image_size": 28,
              "classes": [str(i) for i in range(10)], "official_sizes": (60000, 10000)},
    "fashion_mnist": {"cls": tv_datasets.FashionMNIST, "channels": 1, "image_size": 28,
                      "classes": FASHION_MNIST_CLASSES, "official_sizes": (60000, 10000)},
    "cifar10": {"cls": tv_datasets.CIFAR10, "channels": 3, "image_size": 32,
                "classes": CIFAR10_CLASSES, "official_sizes": (50000, 10000)},
}


def dataset_info(name: str) -> dict:
    if name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASET_INFO)}")
    return DATASET_INFO[name]


# --------------------------------------------------------------------------- raw data
def load_raw(name: str, raw_root, train: bool, download: bool = False):
    """Return (images uint8 [N, C, H, W], labels int64 [N]) of an official split."""
    info = dataset_info(name)
    root = resolve_path(raw_root)
    try:
        ds = info["cls"](root=str(root), train=train, download=download)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not load '{name}' from {root}. "
            f"Run: python scripts/download_data.py --datasets {name}"
        ) from exc
    data = ds.data
    if isinstance(data, np.ndarray):          # CIFAR-10: numpy (N, H, W, C)
        images = torch.from_numpy(data)
        images = images.unsqueeze(1) if images.ndim == 3 else images.permute(0, 3, 1, 2)
    else:                                     # (Fashion-)MNIST: tensor (N, H, W)
        images = data.unsqueeze(1) if data.ndim == 3 else data
    images = images.contiguous().to(torch.uint8)
    labels = torch.as_tensor(np.asarray(ds.targets), dtype=torch.long)
    return images, labels


# --------------------------------------------------------------------------- split
def split_file(split_dir, name: str, split_seed: int, val_fraction: float):
    return resolve_path(split_dir) / f"{name}_split{split_seed}_val{val_fraction:g}.json"


def _channel_stats(images: torch.Tensor, chunk: int = 10000):
    """Per-channel mean/std of uint8 images scaled to [0, 1] (float64, chunked)."""
    c = images.shape[1]
    s = torch.zeros(c, dtype=torch.float64)
    s2 = torch.zeros(c, dtype=torch.float64)
    n = 0
    for i in range(0, images.shape[0], chunk):
        x = images[i:i + chunk].to(torch.float64) / 255.0
        s += x.sum(dim=(0, 2, 3))
        s2 += (x ** 2).sum(dim=(0, 2, 3))
        n += x.shape[0] * x.shape[2] * x.shape[3]
    mean = s / n
    std = torch.sqrt(torch.clamp(s2 / n - mean ** 2, min=1e-12))
    return mean.tolist(), std.tolist()


def _fingerprint(indices) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()[:16]


def create_split(name: str, raw_root, split_dir, split_seed: int = 42, val_fraction: float = 0.1):
    info = dataset_info(name)
    images, labels = load_raw(name, raw_root, train=True)
    _, test_labels = load_raw(name, raw_root, train=False)
    num_classes = len(info["classes"])

    all_idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        all_idx, test_size=val_fraction, random_state=split_seed,
        shuffle=True, stratify=labels.numpy(),
    )
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    mean, std = _channel_stats(images[torch.from_numpy(train_idx)])

    y = labels.numpy()
    payload = {
        "dataset": name,
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "stratified": True,
        "split_unit": "image",
        "protocol": "official train set -> stratified train/val; official test set -> test",
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_labels)),
        "counts": {
            "train": np.bincount(y[train_idx], minlength=num_classes).tolist(),
            "val": np.bincount(y[val_idx], minlength=num_classes).tolist(),
            "test": np.bincount(test_labels.numpy(), minlength=num_classes).tolist(),
        },
        "normalization": {"mean": mean, "std": std, "computed_on": "train split only, pixels scaled to [0,1]"},
        "fingerprints": {"train": _fingerprint(train_idx), "val": _fingerprint(val_idx)},
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
    }
    path = split_file(split_dir, name, split_seed, val_fraction)
    save_json(payload, path, indent=None)
    return payload, path


def get_split(data_cfg: dict, create_if_missing: bool = True, logger=None):
    name = data_cfg["name"]
    split_seed = int(data_cfg.get("split_seed", 42))
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    path = split_file(data_cfg["split_dir"], name, split_seed, val_fraction)
    if path.exists():
        payload = load_json(path)
    else:
        if not create_if_missing:
            raise FileNotFoundError(f"Split file missing: {path}. Run scripts/make_splits.py")
        if logger:
            logger.info("Split file not found, creating %s", path)
        payload, path = create_split(name, data_cfg["raw_root"], data_cfg["split_dir"], split_seed, val_fraction)
    if payload["dataset"] != name:
        raise ValueError(f"Split file {path} belongs to {payload['dataset']}, expected {name}")
    return payload, path


# --------------------------------------------------------------------------- dataset
def augment_image(x: torch.Tensor, random_crop_padding: int = 0, hflip: bool = False) -> torch.Tensor:
    """x: float tensor (C, H, W) in [0, 1]. Zero padding + random crop, then random h-flip."""
    if random_crop_padding and int(random_crop_padding) > 0:
        p = int(random_crop_padding)
        _, h, w = x.shape
        padded = F.pad(x, (p, p, p, p), mode="constant", value=0.0)
        top = int(torch.randint(0, 2 * p + 1, (1,)).item())
        left = int(torch.randint(0, 2 * p + 1, (1,)).item())
        x = padded[:, top:top + h, left:left + w]
    if hflip and torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[2])
    return x


class ImageClassificationDataset(Dataset):
    """Holds uint8 images in memory and applies preprocessing lazily per sample."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, mean, std, augment: dict | None = None):
        self.images = images
        self.labels = labels
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        aug = dict(augment or {})
        active = bool(aug.get("random_crop_padding", 0)) or bool(aug.get("hflip", False))
        self.augment = {"random_crop_padding": int(aug.get("random_crop_padding", 0) or 0),
                        "hflip": bool(aug.get("hflip", False))} if active else None

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        x = self.images[idx].to(torch.float32).div_(255.0)
        if self.augment is not None:
            x = augment_image(x, **self.augment)
        x = (x - self.mean) / self.std
        return x, self.labels[idx]


def build_datasets(cfg: dict, logger=None):
    dcfg = cfg["data"]
    name = dcfg["name"]
    info = dataset_info(name)
    split, split_path = get_split(dcfg, logger=logger)

    trainval_images, trainval_labels = load_raw(name, dcfg["raw_root"], train=True)
    test_images, test_labels = load_raw(name, dcfg["raw_root"], train=False)
    tr = torch.as_tensor(split["train_indices"], dtype=torch.long)
    va = torch.as_tensor(split["val_indices"], dtype=torch.long)
    if len(tr) + len(va) != len(trainval_labels):
        raise ValueError("Split indices do not cover the official training set; recreate the split.")

    mean, std = split["normalization"]["mean"], split["normalization"]["std"]
    datasets = {
        "train": ImageClassificationDataset(trainval_images[tr], trainval_labels[tr], mean, std,
                                            augment=dcfg.get("augment")),
        "val": ImageClassificationDataset(trainval_images[va], trainval_labels[va], mean, std),
        "test": ImageClassificationDataset(test_images, test_labels, mean, std),
    }
    meta = {
        "dataset": name,
        "classes": info["classes"],
        "num_classes": len(info["classes"]),
        "in_channels": info["channels"],
        "image_size": info["image_size"],
        "split_file": str(split_path),
        "split_fingerprints": split["fingerprints"],
        "sizes": {k: len(v) for k, v in datasets.items()},
        "normalization": split["normalization"],
    }
    return datasets, meta


def build_dataloaders(cfg: dict, seed: int, device: torch.device, logger=None):
    datasets, meta = build_datasets(cfg, logger=logger)
    nw = int(cfg["data"].get("num_workers", 0))
    common = {
        "num_workers": nw,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker if nw > 0 else None,
        "persistent_workers": nw > 0,
    }
    g = torch.Generator()
    g.manual_seed(seed)
    bs = int(cfg["train"]["batch_size"])
    ebs = int(cfg["train"].get("eval_batch_size", bs))
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=bs, shuffle=True, drop_last=True,
                            generator=g, **common),
        "val": DataLoader(datasets["val"], batch_size=ebs, shuffle=False, **common),
        "test": DataLoader(datasets["test"], batch_size=ebs, shuffle=False, **common),
    }
    return loaders, datasets, meta
