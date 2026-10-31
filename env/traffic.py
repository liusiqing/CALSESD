"""Shared traffic-transition utility."""

from __future__ import annotations

import numpy as np
import torch


def generate_traffic_matrix(
    current_traffic_matrix,
    baseline_distance_matrix,
    beta: float = 0.07,
    noise_std: float = 1.0,
    trend: float = 5.0,
    generator: torch.Generator | None = None,
    mode: str = "additive",
    ratio: float = 0.3,
):
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1].")
    if mode not in ("additive", "relative") or ratio < 0:
        raise ValueError("invalid traffic trend mode/ratio.")
    current = torch.as_tensor(current_traffic_matrix, dtype=torch.float32, device="cpu")
    baseline = torch.as_tensor(baseline_distance_matrix, dtype=torch.float32, device="cpu")
    if current.shape != baseline.shape or current.ndim != 2 or current.shape[0] != current.shape[1]:
        raise ValueError("Traffic and baseline matrices must be equal square matrices.")
    noise = torch.randn(current.shape, generator=generator, dtype=torch.float32) * noise_std
    if mode == "additive":
        updated = beta * current + (1.0 - beta) * baseline + noise + trend
    else:  # 稳态均值 = baseline * (1 + ratio),短边仍然保持"短"
        updated = beta * current + (1.0 - beta) * baseline * (1.0 + ratio) + noise
    updated = torch.maximum(updated, baseline)
    updated.fill_diagonal_(0.0)
    return updated.numpy().astype(np.float64, copy=False)
