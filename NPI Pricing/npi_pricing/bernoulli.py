"""NPI for Bernoulli data — the model-free baseline.

Given ``n`` past windows of length equal to the contract horizon, of which ``s``
crossed the boundary, Nonparametric Predictive Inference gives the lower/upper
probability that the *next* window crosses:

    P_lower = s / (n + 1)
    P_upper = (s + 1) / (n + 1)

(Coolen, 1998 — NPI for Bernoulli quantities, derived from Hill's A(n).)

This ignores *where* the price currently sits within a window and treats every
window as exchangeable, so it is deliberately crude. Its virtue is that it makes
no path or distributional assumption whatsoever: it just counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NPIBernoulli:
    n: int          # number of past trials (windows)
    s: int          # number of successes (crossings)
    lower: float    # NPI lower probability for the next trial
    upper: float    # NPI upper probability for the next trial

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def imprecision(self) -> float:
        """Width of the probability interval (== 1 / (n + 1))."""
        return self.upper - self.lower


def npi_bernoulli(successes: int, trials: int) -> NPIBernoulli:
    """NPI lower/upper probability that the next trial is a success."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    lower = successes / (trials + 1)
    upper = (successes + 1) / (trials + 1)
    return NPIBernoulli(n=trials, s=successes, lower=lower, upper=upper)


def _running_extreme_log_return(window_prices: np.ndarray, *, up: bool) -> float:
    """Most extreme cumulative log-return reached *within* a window, from its start."""
    log_ret = np.log(window_prices / window_prices[0])
    return float(log_ret.max() if up else log_ret.min())


def bernoulli_crossing_probability(
    prices: np.ndarray,
    spot: float,
    barrier: float,
    horizon: int,
    *,
    up: bool | None = None,
    stride: int = 1,
) -> NPIBernoulli:
    """NPI-Bernoulli probability of touching ``barrier`` within ``horizon`` steps.

    Historical windows of length ``horizon`` are scored for whether their running
    extreme log-return reaches the *relative* move implied by the current spot and
    barrier (``log(barrier / spot)``). This conditions the count on today's
    distance to the barrier, which is what makes the windows comparable.

    Parameters
    ----------
    prices : 1-D array of historical prices (chronological).
    spot : current underlying price.
    barrier : the boundary level.
    horizon : contract length, in the same step units as ``prices``.
    up : force an up-barrier (True) or down-barrier (False); inferred from
         ``barrier`` vs ``spot`` when None.
    stride : window start spacing. ``stride=1`` overlaps windows (more "trials"
             but heavily autocorrelated); ``stride=horizon`` gives independent,
             non-overlapping windows (honest ``n``, fewer trials).
    """
    prices = np.asarray(prices, dtype=float)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if prices.size < horizon + 1:
        raise ValueError("not enough history for one full window")
    if up is None:
        up = barrier >= spot

    target = np.log(barrier / spot)  # relative move to the barrier

    successes = 0
    trials = 0
    last_start = prices.size - (horizon + 1)
    for start in range(0, last_start + 1, stride):
        window = prices[start : start + horizon + 1]
        extreme = _running_extreme_log_return(window, up=up)
        crossed = extreme >= target if up else extreme <= target
        successes += int(crossed)
        trials += 1

    return npi_bernoulli(successes, trials)
