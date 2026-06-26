"""Shared records that cross module boundaries. Lightweight by design."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import numpy as np
import pandas as pd


@dataclass
class OptionChain:
    """Cleaned Deribit option chain for a single expiry.

    `quotes` columns: instrument_name, strike, option_type, bid_iv, ask_iv,
    mid_iv (decimal), best_bid_price, best_ask_price, mid_price (USD),
    open_interest, delta. IV is stored in DECIMAL form (0.8 == 80%).
    """
    expiry_timestamp: int          # ms epoch
    underlying_price: float
    fetch_time: int                # s epoch
    quotes: pd.DataFrame

    @property
    def time_to_expiry(self) -> float:
        """Years until expiry, from fetch_time."""
        return max((self.expiry_timestamp / 1000 - self.fetch_time) / (365.0 * 24 * 3600), 0.0)


@dataclass
class RNDResult:
    """Risk-neutral density on a strike grid."""
    strikes: np.ndarray
    density: np.ndarray
    expiry_timestamp: int
    underlying_price: float
    diagnostics: dict = field(default_factory=dict)


@dataclass
class PolymarketContract:
    """A parsed Polymarket BTC price-prediction market."""
    contract_id: str
    question: str
    bet_type: str                  # "above" | "range" | "reach" | "dip" | "UpDown" | "below"
    strike_low: Optional[float]
    strike_high: Optional[float]
    resolution_timestamp: int      # s epoch
    polymarket_price: Optional[float]
    contract_lifetime: Optional[tuple[datetime, datetime]]


@dataclass
class PricingResult:
    """RND-implied fair value vs Polymarket price for one contract."""
    contract_id: str
    bet_type: str
    polymarket_price: Optional[float]
    rnd_fair_value: Optional[float]
    deviation: Optional[float]
    band_lower: Optional[float] = None
    band_upper: Optional[float] = None
    violation: Optional[str] = None
    note: str = ""
