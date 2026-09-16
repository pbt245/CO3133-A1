"""Inference-time measurement (forward pass only, data already on the device)."""
from __future__ import annotations

import time

import numpy as np
import torch

from src.engine.loops import autocast_ctx
from src.utils.env import synchronize


@torch.no_grad()
def measure_inference(model, dataset, device, batch_size: int = 256, warmup_iters: int = 10,
                      timed_iters: int = 50, latency_samples: int = 200, amp: bool = False) -> dict:
    """Two views of inference cost:
      * throughput: images/s for repeated forward passes of one fixed batch
      * latency   : ms per single image (batch size 1), mean / median / p95
    Warm-up iterations are excluded; GPU is synchronized before reading the clock."""
    model.eval()
    n_needed = min(max(int(batch_size), int(latency_samples)), len(dataset))
    samples = torch.stack([dataset[i][0] for i in range(n_needed)]).to(device)
    bs = min(int(batch_size), n_needed)
    batch = samples[:bs]
    use_amp = bool(amp) and device.type == "cuda"

    with autocast_ctx(device, use_amp):
        for _ in range(int(warmup_iters)):
            model(batch)
        synchronize(device)
        t0 = time.perf_counter()
        iters = max(int(timed_iters), 1)
        for _ in range(iters):
            model(batch)
        synchronize(device)
        elapsed = time.perf_counter() - t0

        m = min(int(latency_samples), n_needed)
        for i in range(min(20, m)):
            model(samples[i:i + 1])
        latencies = []
        for i in range(m):
            synchronize(device)
            t = time.perf_counter()
            model(samples[i:i + 1])
            synchronize(device)
            latencies.append((time.perf_counter() - t) * 1000.0)

    lat = np.asarray(latencies) if latencies else np.asarray([np.nan])
    return {
        "device": str(device),
        "amp": use_amp,
        "torch_num_threads": torch.get_num_threads(),
        "batch_size": bs,
        "timed_iters": iters,
        "throughput_img_per_s": float(bs * iters / elapsed),
        "ms_per_image_batched": float(1000.0 * elapsed / (bs * iters)),
        "latency_samples": int(m),
        "latency_ms_bs1_mean": float(np.mean(lat)),
        "latency_ms_bs1_median": float(np.median(lat)),
        "latency_ms_bs1_p95": float(np.percentile(lat, 95)),
    }
