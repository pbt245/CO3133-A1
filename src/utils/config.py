"""YAML config loading with `base:` inheritance and `key.sub=value` overrides."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path) -> dict:
    """Load a YAML file. A `base:` key (str or list) is resolved relative to that file
    and merged first; values in the current file win."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    bases = cfg.pop("base", None)
    if bases:
        if isinstance(bases, str):
            bases = [bases]
        merged: dict = {}
        for b in bases:
            merged = _deep_merge(merged, load_config(path.parent / b))
        cfg = _deep_merge(merged, cfg)
    return cfg


def apply_overrides(cfg: dict, overrides) -> dict:
    """Apply overrides such as ["train.epochs=3", "data.augment.hflip=false"]."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got '{item}'")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
    return cfg


def save_config(cfg: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
