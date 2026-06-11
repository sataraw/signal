"""Black-Scholes from scratch. sigma is a decimal (0.8 == 80%); T in years."""
from __future__ import annotations

from typing import Optional

import numpy as np


def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    raise NotImplementedError("M3")


def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    raise NotImplementedError("M3")


def implied_vol(market_price: float, S: float, K: float, T: float, r: float,
                option_type: str = "call") -> Optional[float]:
    raise NotImplementedError("M3")


def vectorized_call_price(S: float, K_array: np.ndarray, T: float, r: float,
                          sigma_array: np.ndarray) -> np.ndarray:
    raise NotImplementedError("M3")
