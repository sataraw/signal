"""Price Polymarket contracts against an extracted RND."""
from __future__ import annotations

import numpy as np
import pandas as pd

from schemas import PolymarketContract, PricingResult, RNDResult


def price_above(rnd: RNDResult, strike: float) -> float:
    """P(S_T >= strike)."""
    raise NotImplementedError("M5")


def price_range(rnd: RNDResult, low: float, high: float) -> float:
    """P(low <= S_T <= high)."""
    raise NotImplementedError("M5")


def price_contract(contract: PolymarketContract, rnd: RNDResult) -> PricingResult:
    raise NotImplementedError("M5")


def price_all_contracts(contracts: list[PolymarketContract], rnd: RNDResult) -> pd.DataFrame:
    raise NotImplementedError("M5")
