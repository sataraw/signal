"""Synthetic sample data so the prototype runs out of the box."""

from __future__ import annotations

import numpy as np


def make_sample_prices(
    n_days: int = 750,
    s0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.20,
    dt: float = 1.0 / 252.0,
    seed: int | None = 7,
) -> np.ndarray:
    """A geometric-Brownian-motion daily price series (annualised mu/sigma)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_days)
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    return s0 * np.exp(np.concatenate([[0.0], np.cumsum(log_ret)]))
