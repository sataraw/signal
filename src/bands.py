"""No-arbitrage band construction and violation detection (Stage 5 / M6)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import GAS_COST_USD, POLYMARKET_FEE
from schemas import PolymarketContract, PricingResult, RNDResult


def call_spread_bounds(rnd: RNDResult, strike: float,
                       deribit_strikes: np.ndarray) -> tuple[float, float]:
    raise NotImplementedError("M6")


def transaction_cost_adjustment(num_options: int = 4) -> float:
    raise NotImplementedError("M6")


def compute_band(contract: PolymarketContract, rnd: RNDResult,
                 deribit_strikes: np.ndarray) -> tuple[float, float]:
    raise NotImplementedError("M6")


def check_violation(pricing_result: PricingResult) -> str | None:
    raise NotImplementedError("M6")


def run_band_analysis(results: pd.DataFrame, rnd: RNDResult,
                      deribit_strikes: np.ndarray) -> pd.DataFrame:
    raise NotImplementedError("M6")
