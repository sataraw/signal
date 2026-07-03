"""Arbitrage / deviation bound for a one-touch contract.

The NPI lattice already returns an imprecise probability bracket [L, U] that
captures *model ambiguity* (worst/best case within the A(n) gap structure). But
to call a market price "mispriced enough to trade" we must also account for:

  1. Estimation noise — [L, U] is computed from one finite sample of historical
     increments. A block bootstrap (which preserves volatility clustering and
     autocorrelation, our main model-risk caveat) gives the sampling spread.
  2. Discrete-vs-continuous monitoring — Polymarket resolves on the live exchange
     price (effectively continuous), but a daily-step lattice only checks once a
     day. The Broadie-Glasserman-Kou continuity correction shifts the barrier so
     a fast daily engine reproduces continuous-monitoring touch probabilities.
  3. Trading frictions — the bid/ask half-spread and any fee.

The resulting bound is:

    arb_lower = quantile_alpha( L over bootstraps )  - friction
    arb_upper = quantile_{1-alpha}( U over bootstraps ) + friction

A market YES price below arb_lower is an underpricing (BUY YES); above arb_upper
is an overpricing (SELL YES). Inside the band, the disagreement is within what
ambiguity + sampling noise + frictions can explain, so there is no actionable
edge. The confidence level (1 - 2*alpha) controls the false-positive rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .first_passage import FirstPassageResult, ImpreciseLattice, grid_bounds

# Broadie-Glasserman-Kou continuity-correction constant: -zeta(1/2)/sqrt(2*pi).
_BGK_BETA = 0.5826


def continuity_corrected_barrier(
    barrier_log: float, sigma_step: float, *, up: bool
) -> float:
    """Shift a continuously-monitored barrier inward for a per-step-monitored engine.

    Discrete monitoring misses touches between observations and so understates the
    touch probability; moving the barrier toward spot by beta*sigma*sqrt(dt)
    (dt = 1 step) compensates (Broadie-Glasserman-Kou, 1997).
    """
    shift = _BGK_BETA * sigma_step
    return barrier_log - shift if up else barrier_log + shift


def _block_bootstrap(rng: np.random.Generator, x: np.ndarray, block: int) -> np.ndarray:
    """Moving-block bootstrap resample of a 1-D series (preserves local dependence)."""
    n = x.size
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
    return x[idx]


@dataclass
class ConfidenceBand:
    npi_lower: float          # full-sample NPI lower touch prob
    npi_upper: float          # full-sample NPI upper touch prob
    arb_lower: float          # actionable lower bound (with frictions)
    arb_upper: float          # actionable upper bound (with frictions)
    friction: float
    alpha: float
    n_boot: int

    @property
    def width(self) -> float:
        return self.arb_upper - self.arb_lower

    def classify(self, market_price: float) -> tuple[str, float]:
        """Return (signal, edge). Edge is the distance outside the band (>=0)."""
        if market_price < self.arb_lower:
            return "BUY YES", self.arb_lower - market_price
        if market_price > self.arb_upper:
            return "SELL YES", market_price - self.arb_upper
        return "no-trade", 0.0


def _semi_deviations(x: np.ndarray) -> tuple[float, float]:
    """(upside, downside) root-mean-square semi-deviations of a zero-ref series."""
    up, dn = x[x > 0], x[x < 0]
    su = float(np.sqrt(np.mean(up**2))) if up.size else 0.0
    sd = float(np.sqrt(np.mean(dn**2))) if dn.size else 0.0
    return su, sd


def _regime_adjust(
    increments: np.ndarray,
    vol_window: int | None,
    *,
    skew_aware: bool = True,
    drift: str = "zero",
) -> np.ndarray:
    """Regime-match historical increments, optionally with separate up/down vol.

    Volatility-adjusted historical simulation (Hull-White, 1998), extended to be
    *skew-aware*: the upside and downside residuals are rescaled by their OWN
    recent/full semi-deviation ratios, so a long sample whose up-tail (e.g. a past
    bull run) is fatter than today's gets its up-moves shrunk independently of the
    down-moves. Drift is reset to the chosen regime so a stale historical trend no
    longer inflates one-sided touch probabilities.

      drift = "recent" (current-regime mean) | "zero" (driftless) | "full" (legacy)

    The empirical *shape* within each tail (the NPI gap structure) is preserved;
    only each tail's scale and the overall drift are retargeted.
    """
    r = np.asarray(increments, dtype=float)
    if not vol_window or r.size <= vol_window:
        return r

    full_mean = float(r.mean())
    recent = r[-vol_window:]
    recent_mean = float(recent.mean())
    mu = {"recent": recent_mean, "zero": 0.0, "full": full_mean}.get(drift)
    if mu is None:
        raise ValueError("drift must be 'recent', 'zero' or 'full'")

    e = r - full_mean  # residuals carry the distribution shape
    if skew_aware:
        full_up, full_dn = _semi_deviations(e)
        rec_up, rec_dn = _semi_deviations(recent - recent_mean)
        s_up = (rec_up / full_up) if full_up > 0 else 1.0
        s_dn = (rec_dn / full_dn) if full_dn > 0 else 1.0
        scaled = np.where(e > 0, e * s_up, e * s_dn)
    else:
        full_sd = float(np.std(e))
        rec_sd = float(np.std(recent - recent_mean))
        scaled = e * ((rec_sd / full_sd) if full_sd > 0 else 1.0)

    # set the drift exactly to mu; skew scaling only affects dispersion/shape
    return mu + (scaled - float(scaled.mean()))


def npi_probability(
    prices: np.ndarray,
    spot: float,
    barrier: float,
    horizon: int,
    *,
    up: bool | None = None,
    kind: str = "touch",
    n_states: int = 121,
    continuous_monitoring: bool = True,
    vol_window: int | None = 30,
    skew_aware: bool = True,
    drift: str = "zero",
) -> FirstPassageResult:
    """Full-sample NPI physical-probability bracket [L, U] for a contract.

    Same modelling as ``arbitrage_band`` (regime adjustment + continuity
    correction) but a single point estimate with no bootstrap — this is "our
    price". ``skew_aware``/``drift`` control the regime adjustment (see
    ``_regime_adjust``).
    """
    prices = np.asarray(prices, dtype=float)
    increments = _regime_adjust(
        np.diff(np.log(prices)), vol_window, skew_aware=skew_aware, drift=drift
    )
    if up is None:
        up = barrier >= spot
    spot_log, barrier_log = np.log(spot), np.log(barrier)
    if kind == "touch" and continuous_monitoring:
        barrier_log = continuity_corrected_barrier(
            barrier_log, float(np.std(increments)), up=up
        )
    lat = ImpreciseLattice.from_history(
        increments, spot_log, barrier_log, horizon, n_states=n_states
    )
    if kind == "terminal":
        return lat.terminal(spot_log, barrier_log, horizon, up=up)
    return lat.first_passage(spot_log, barrier_log, horizon, up=up)


def arbitrage_band(
    prices: np.ndarray,
    spot: float,
    barrier: float,
    horizon: int,
    *,
    up: bool | None = None,
    kind: str = "touch",
    n_states: int = 121,
    alpha: float = 0.05,
    n_boot: int = 200,
    block: int | None = None,
    friction: float = 0.0,
    continuous_monitoring: bool = True,
    vol_window: int | None = 30,
    skew_aware: bool = True,
    drift: str = "zero",
    seed: int | None = 12345,
) -> ConfidenceBand:
    """Bootstrap confidence band / arbitrage bound for a contract.

    ``kind`` is "touch" (one-touch barrier, path-dependent) or "terminal"
    (European digital, endpoint only). Continuity correction applies only to
    touch contracts. ``vol_window`` regime-matches history (None disables);
    ``skew_aware``/``drift`` control the up/down-tail rescaling and drift reset.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.size < 5:
        raise ValueError("need more price history")
    increments = _regime_adjust(
        np.diff(np.log(prices)), vol_window, skew_aware=skew_aware, drift=drift
    )
    if up is None:
        up = barrier >= spot

    spot_log = np.log(spot)
    barrier_log = np.log(barrier)
    if kind == "touch" and continuous_monitoring:
        sigma_step = float(np.std(increments))
        barrier_log = continuity_corrected_barrier(barrier_log, sigma_step, up=up)

    # Fix the grid from the full sample so every bootstrap is comparable.
    lo, hi = grid_bounds(increments, spot_log, barrier_log, horizon)

    def npi_bracket(incr: np.ndarray) -> tuple[float, float]:
        lat = ImpreciseLattice(incr, lo, hi, n_states)
        if kind == "terminal":
            r = lat.terminal(spot_log, barrier_log, horizon, up=up)
        else:
            r = lat.first_passage(spot_log, barrier_log, horizon, up=up)
        return r.lower, r.upper

    full_lo, full_hi = npi_bracket(increments)

    rng = np.random.default_rng(seed)
    block = block or max(2, int(round(np.sqrt(increments.size))))
    lows = np.empty(n_boot)
    highs = np.empty(n_boot)
    for b in range(n_boot):
        resample = _block_bootstrap(rng, increments, block)
        lows[b], highs[b] = npi_bracket(resample)

    arb_lower = float(np.percentile(lows, 100 * alpha)) - friction
    arb_upper = float(np.percentile(highs, 100 * (1 - alpha))) + friction

    return ConfidenceBand(
        npi_lower=full_lo,
        npi_upper=full_hi,
        arb_lower=float(np.clip(arb_lower, 0.0, 1.0)),
        arb_upper=float(np.clip(arb_upper, 0.0, 1.0)),
        friction=friction,
        alpha=alpha,
        n_boot=n_boot,
    )
