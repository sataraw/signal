"""Fetch BTC price history and spot to feed the NPI engine (Binance public API)."""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime

import numpy as np

_HEADERS = {"User-Agent": "Mozilla/5.0 (npi-pricing research)"}
# Binance has several public hostnames; we try them in order for resilience.
_HOSTS = ("https://api.binance.com", "https://data-api.binance.vision")


def _get(path: str) -> object:
    last_err: Exception | None = None
    for host in _HOSTS:
        try:
            req = urllib.request.Request(host + path, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except Exception as e:  # noqa: BLE001 - try next host
            last_err = e
    raise RuntimeError(f"all Binance hosts failed for {path}: {last_err}")


def binance_closes(symbol: str = "BTCUSDT", interval: str = "1d", limit: int = 1000) -> np.ndarray:
    """Closing prices (chronological). ``interval`` e.g. '1d', '1h'."""
    raw = _get(f"/api/v3/klines?symbol={symbol}&interval={interval}&limit={int(limit)}")
    return np.array([float(k[4]) for k in raw], dtype=float)


def binance_spot(symbol: str = "BTCUSDT") -> float:
    """Current price."""
    raw = _get(f"/api/v3/ticker/price?symbol={symbol}")
    return float(raw["price"])


def binance_price_at(when: datetime, symbol: str = "BTCUSDT") -> float | None:
    """Open of the 1-minute kline starting at ``when`` (tz-aware, UTC).

    Used to recover the strike of Polymarket Up/Down markets (the reference
    price at window start), which the Polymarket API does not expose.
    Returns None if Binance has no kline for that minute.
    """
    start_ms = int(when.timestamp() * 1000)
    raw = _get(f"/api/v3/klines?symbol={symbol}&interval=1m&startTime={start_ms}&limit=1")
    if not raw:
        return None
    kline = raw[0]
    # Guard against Binance returning the next available kline for far-past/future times.
    if abs(int(kline[0]) - start_ms) > 60_000:
        return None
    return float(kline[1])
