"""Fetch and parse current Bitcoin one-touch contracts from Polymarket (Gamma API).

Polymarket phrases these as:
  "Will Bitcoin reach $X in <month>?"   -> one-touch UP barrier at X
  "Will Bitcoin dip to $X in <month>?"  -> one-touch DOWN barrier at X
The YES outcome price is the market-implied touch probability.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

_HEADERS = {"User-Agent": "Mozilla/5.0 (npi-pricing research)"}
_GAMMA = "https://gamma-api.polymarket.com"

# one-touch (path-dependent: crossed at any time in the window)
_TOUCH_UP = ("reach", "hit", "climb to", "rise to", "touch", "surpass")
_TOUCH_DOWN = ("dip", "drop to", "fall to", "down to")
# terminal (European digital: condition holds at the resolution date)
_TERM_UP = ("above", "exceed", "higher than", "over ")
_TERM_DOWN = ("below", "under ", "lower than")
_PRICE_RE = re.compile(r"\$\s?([\d][\d,]*(?:\.\d+)?)\s?([kKmM]?)")


@dataclass
class Contract:
    question: str
    barrier: float
    up: bool                 # True = upward condition, False = downward
    kind: str                # "touch" (one-touch barrier) or "terminal" (digital)
    end: datetime
    yes_price: float         # market mid (outcomePrices[0])
    bid: float
    ask: float
    liquidity: float
    volume: float
    slug: str

    def horizon_days(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        return (self.end - now).total_seconds() / 86400.0

    @property
    def half_spread(self) -> float:
        if self.ask > 0 and self.bid > 0 and self.ask >= self.bid:
            return 0.5 * (self.ask - self.bid)
        return 0.0


def _get(url: str) -> object:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def _parse_barrier(question: str) -> float | None:
    m = _PRICE_RE.search(question)
    if not m:
        return None
    val = float(m.group(1).replace(",", ""))
    suffix = m.group(2).lower()
    if suffix == "k":
        val *= 1_000
    elif suffix == "m":
        val *= 1_000_000
    return val


def _parse_kind_direction(question: str) -> tuple[str, bool] | None:
    """Return (kind, up). Touch words win over terminal words when both appear."""
    q = question.lower()
    if any(w in q for w in _TOUCH_DOWN):
        return "touch", False
    if any(w in q for w in _TOUCH_UP):
        return "touch", True
    if any(w in q for w in _TERM_DOWN):
        return "terminal", False
    if any(w in q for w in _TERM_UP):
        return "terminal", True
    return None


def _to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _market_to_contract(mk: dict) -> Contract | None:
    q = mk.get("question", "")
    barrier = _parse_barrier(q)
    kd = _parse_kind_direction(q)
    if barrier is None or kd is None or not mk.get("endDate"):
        return None
    kind, up = kd
    try:
        prices = json.loads(mk.get("outcomePrices") or "[]")
        yes = float(prices[0]) if prices else float(mk.get("lastTradePrice", 0))
    except (ValueError, TypeError, IndexError):
        return None
    return Contract(
        question=q,
        barrier=barrier,
        up=up,
        kind=kind,
        end=_to_dt(mk["endDate"]),
        yes_price=yes,
        bid=float(mk.get("bestBid") or 0.0),
        ask=float(mk.get("bestAsk") or 0.0),
        liquidity=float(mk.get("liquidity") or 0.0),
        volume=float(mk.get("volume") or 0.0),
        slug=mk.get("slug", ""),
    )


def fetch_btc_contracts(
    *,
    min_price: float = 0.02,
    max_price: float = 0.98,
    min_liquidity: float = 0.0,
) -> list[Contract]:
    """Live BTC one-touch contracts whose YES price is informative (not ~0 or ~1).

    ``min_price``/``max_price`` drop already-resolved / dead markets so we only
    test the pricer on contracts the market still treats as uncertain.
    """
    events = _get(f"{_GAMMA}/public-search?q=bitcoin&limit_per_type=40").get("events", [])
    out: dict[str, Contract] = {}
    for ev in events:
        if ev.get("closed") or not ev.get("active"):
            continue
        for mk in ev.get("markets", []):
            if mk.get("closed") or not mk.get("active"):
                continue
            c = _market_to_contract(mk)
            if c is None:
                continue
            if not (min_price <= c.yes_price <= max_price):
                continue
            if c.liquidity < min_liquidity:
                continue
            if c.horizon_days() <= 0:
                continue
            out[c.slug or c.question] = c  # dedupe
    return sorted(out.values(), key=lambda c: (c.up, c.barrier))
