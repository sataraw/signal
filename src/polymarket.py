"""Polymarket fetching: Gamma API for discovery + question parsing, CLOB for live prices."""
from __future__ import annotations

from typing import Optional

from schemas import PolymarketContract


def get_btc_contracts() -> list[PolymarketContract]:
    """Active BTC price-prediction markets, parsed into PolymarketContract."""
    raise NotImplementedError("M2")


def parse_strike_from_question(question: str) -> tuple[Optional[float], Optional[float]]:
    """Extract strike(s) from question text. (strike, None) for above/reach/dip,
    (low, high) for range, (None, None) on failure."""
    raise NotImplementedError("M2")
