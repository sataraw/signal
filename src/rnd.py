"""Risk-neutral density extraction: IV smile fit -> tail extrapolation ->
Breeden-Litzenberger (2nd derivative of call price wrt strike)."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from config import RISK_FREE_RATE
from schemas import OptionChain, RNDResult


def fit_iv_smile(strikes: np.ndarray, mid_ivs: np.ndarray) -> CubicSpline:
    raise NotImplementedError("M4")


def extrapolate_tails(strikes: np.ndarray, ivs: np.ndarray, S: float, T: float,
                      r: float) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("M4")


def iv_smile_to_call_prices(strike_grid: np.ndarray, iv_spline: CubicSpline, S: float,
                            T: float, r: float) -> np.ndarray:
    raise NotImplementedError("M4")


def breeden_litzenberger(strike_grid: np.ndarray, call_prices: np.ndarray, r: float,
                         T: float) -> RNDResult:
    raise NotImplementedError("M4")


def extract_rnd(chain: OptionChain, r: float = RISK_FREE_RATE) -> RNDResult:
    """Top-level: cleaned chain -> RNDResult."""
    raise NotImplementedError("M4")
