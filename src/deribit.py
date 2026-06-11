"""All Deribit API communication. Nothing outside this module calls requests for Deribit.

Boundary conventions enforced here so the rest of the codebase never has to think
about them:
  - IV is converted from percent to DECIMAL (54.13 -> 0.5413).
  - Option prices are converted from BTC to USD (price * underlying_price).
"""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from config import DERIBIT_BASE_URL, MIN_STRIKES
from schemas import OptionChain

log = logging.getLogger(__name__)

_SESSION = requests.Session()
_DAY_MS = 86_400_000


def _get(endpoint: str, params: dict) -> dict | None:
    """GET a Deribit public endpoint. Returns the `result` dict, or None on API error.

    Network errors are logged and re-raised; API-level errors are logged and return None.
    """
    url = f"{DERIBIT_BASE_URL}/{endpoint}"
    try:
        resp = _SESSION.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error("Deribit request failed: %s params=%s err=%s", endpoint, params, e)
        raise
    payload = resp.json()
    if "error" in payload:
        log.error("Deribit API error on %s: %s", endpoint, payload["error"])
        return None
    return payload.get("result")


def get_instruments() -> pd.DataFrame:
    """Active BTC option instruments.

    Columns: instrument_name, strike, expiry_timestamp (ms), option_type ('call'/'put').
    """
    result = _get("get_instruments", {"currency": "BTC", "kind": "option", "expired": "false"})
    df = pd.DataFrame(result)
    df = df.rename(columns={"expiration_timestamp": "expiry_timestamp"})
    return df[["instrument_name", "strike", "expiry_timestamp", "option_type"]]


def get_order_book(instrument_name: str) -> dict | None:
    """Raw order book for one instrument, or None if no two-sided market / request fails."""
    result = _get("get_order_book", {"instrument_name": instrument_name, "depth": 1})
    if result is None:
        return None
    if not result.get("bids") or not result.get("asks"):
        return None
    return result


def get_option_chain(expiry_timestamp: int) -> OptionChain | None:
    """Cleaned OptionChain for one expiry. IV in decimals, mid_price in USD.

    Cleaning: drop instruments with zero/null bid_iv or ask_iv, or zero open interest.
    """
    instruments = get_instruments()
    at_expiry = instruments[instruments["expiry_timestamp"] == expiry_timestamp]
    if at_expiry.empty:
        log.warning("No instruments at expiry %s", expiry_timestamp)
        return None

    rows = []
    underlyings = []
    for inst in at_expiry.itertuples():
        ob = get_order_book(inst.instrument_name)
        if ob is None:
            continue

        bid_iv, ask_iv = ob.get("bid_iv"), ob.get("ask_iv")
        if not bid_iv or not ask_iv:            # 0 or None -> no usable market
            continue
        if not ob.get("open_interest"):          # 0 or None
            continue

        underlying = ob["underlying_price"]
        underlyings.append(underlying)
        bid_iv, ask_iv = bid_iv / 100.0, ask_iv / 100.0   # percent -> decimal
        mid_price_btc = (ob["best_bid_price"] + ob["best_ask_price"]) / 2.0
        rows.append({
            "instrument_name": inst.instrument_name,
            "strike": inst.strike,
            "option_type": inst.option_type,
            "bid_iv": bid_iv,
            "ask_iv": ask_iv,
            "mid_iv": (bid_iv + ask_iv) / 2.0,
            "best_bid_price": ob["best_bid_price"],
            "best_ask_price": ob["best_ask_price"],
            "mid_price": mid_price_btc * underlying,    # USD
            "open_interest": ob["open_interest"],
            "delta": ob.get("greeks", {}).get("delta"),
        })

    if not rows:
        log.warning("No instruments survived cleaning at expiry %s", expiry_timestamp)
        return None

    quotes = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    if len(quotes) < MIN_STRIKES:
        log.warning("Only %d strikes survived cleaning at expiry %s (min %d)",
                    len(quotes), expiry_timestamp, MIN_STRIKES)

    return OptionChain(
        expiry_timestamp=int(expiry_timestamp),
        underlying_price=float(pd.Series(underlyings).median()),
        fetch_time=int(time.time()),
        quotes=quotes,
    )


def find_nearest_expiry(target_timestamp: int) -> int:
    """Deribit expiry (ms epoch) closest to target. Logs the mismatch if > 3 days.

    Accepts target in seconds or milliseconds; normalizes to ms.
    """
    target_ms = target_timestamp * 1000 if target_timestamp < 1e12 else target_timestamp
    expiries = get_instruments()["expiry_timestamp"].unique()
    nearest = int(min(expiries, key=lambda e: abs(e - target_ms)))
    mismatch_days = abs(nearest - target_ms) / _DAY_MS
    if mismatch_days > 3:
        log.warning("Nearest Deribit expiry is %.1f days from target (settlement basis risk)",
                    mismatch_days)
    return nearest
