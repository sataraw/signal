"""Wedge tester: compare the *implied* pricing wedge to the *empirical* one.

Yang (2026) models the prediction-market price as a Wang transform of the
physical probability:

    p_mkt = Phi( Phi^{-1}(p*) + lambda )      <=>      lambda = Phi^{-1}(p_mkt) - Phi^{-1}(p*)

Given OUR physical-probability band [L, U] (from the NPI engine) and the market
price, the *implied wedge* is the range of lambda needed to reconcile the two:

    implied_lambda in [ Phi^{-1}(p_mkt) - Phi^{-1}(U) , Phi^{-1}(p_mkt) - Phi^{-1}(L) ].

We then ask whether that range overlaps an *empirically plausible* wedge range.
This is the imprecise-probability-consistent version of "shift by the Wang
transform": rather than committing to a single lambda, we ask whether *any*
plausible lambda explains the price.

  - overlap            -> price is explained by a normal risk premium (no edge)
  - implied entirely above plausible -> market too rich vs our price (SELL YES)
  - implied entirely below plausible -> market too cheap vs our price (BUY YES)

Phi / Phi^{-1} are implemented here (Acklam's inverse-normal approximation) to
avoid a SciPy dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_EPS = 1e-9

# ---------------------------------------------------------------------------
# Standard-normal CDF and inverse CDF (no SciPy)
# ---------------------------------------------------------------------------
_A = (-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
      1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0)
_B = (-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
      6.680131188771972e1, -1.328068155288572e1)
_C = (-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
      -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0)
_D = (7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
      3.754408661907416e0)
_PLOW, _PHIGH = 0.02425, 1 - 0.02425


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam), scalar."""
    p = min(max(p, _EPS), 1.0 - _EPS)
    if p < _PLOW:
        q = math.sqrt(-2.0 * math.log(p))
        return ((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
                / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0))
    if p > _PHIGH:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
                 / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0))
    q = p - 0.5
    r = q * q
    return ((((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q
            / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0))


def wang_transform(p_star: float, lam: float) -> float:
    """Market price implied by physical prob ``p_star`` under wedge ``lam``."""
    return norm_cdf(norm_ppf(p_star) + lam)


# ---------------------------------------------------------------------------
# Implied wedge (from our price + market price)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ImpliedWedge:
    lo: float       # smallest lambda consistent with our band (uses p* = upper)
    mid: float
    hi: float       # largest lambda consistent with our band (uses p* = lower)


def implied_wedge(market_price: float, p_lower: float, p_upper: float) -> ImpliedWedge:
    z_m = norm_ppf(market_price)
    lo = z_m - norm_ppf(p_upper)
    hi = z_m - norm_ppf(p_lower)
    mid = z_m - norm_ppf(0.5 * (p_lower + p_upper))
    return ImpliedWedge(lo=lo, mid=mid, hi=hi)


# ---------------------------------------------------------------------------
# Empirical wedge (from Yang 2026)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmpiricalWedge:
    """A plausible lambda range. Default: Polymarket headline MLE + 95% CI.

    ``shrink`` pulls the range toward 0 (use < 1 for liquid / short-dated /
    extreme-probability crypto contracts, where the paper's determinants imply a
    smaller wedge than the 0.176 cross-platform average).
    """
    point: float = 0.176
    ci: tuple[float, float] = (0.123, 0.230)
    shrink: float = 1.0

    def range(self) -> tuple[float, float]:
        return self.ci[0] * self.shrink, self.ci[1] * self.shrink


# Yang (2026) Table 6, spec (2) slope coefficients. We use them as *tilts around
# a baseline*, not as an absolute formula: the paper's constant and the raw
# ln(Volume)/ln(Duration) terms are unit-sensitive (a $-volume in absolute terms
# pushes lambda by several tenths and dominates the fit), so applying them
# literally is unreliable. Anchoring at the robust headline lambda and adding
# only *relative* covariate deviations keeps the level sane and unit-safe.
_XS = dict(b_vol=-0.057, b_dur=0.109, b_ext=-0.290, g1=-0.156, g2=0.074)
_EXT_REF = 0.25     # mean of |p-0.5| under a flat prior over (0,1)


def cross_sectional_lambda(
    market_price: float,
    *,
    baseline: float = 0.176,
    volume: float | None = None,
    vol_ref: float = 100_000.0,
    duration_days: float | None = None,
    dur_ref: float = 30.0,
    life_fraction: float = 0.0,
    uncertainty: float = 0.08,
) -> EmpiricalWedge:
    """Characteristic-tilted empirical wedge anchored at ``baseline``.

    Starts from the robust headline wedge and applies Yang (2026) Table-6 slopes
    as deviations from a reference contract (so a typical contract returns the
    baseline). Extremity and time-decay are unit-safe; volume/duration are
    relative to ``vol_ref``/``dur_ref`` and contribute 0 at the reference.
    """
    tau = min(max(life_fraction, 0.0), 1.0)
    lam = baseline
    lam += _XS["b_ext"] * (abs(market_price - 0.5) - _EXT_REF)
    lam += _XS["g1"] * tau + _XS["g2"] * tau * tau
    if volume is not None:
        lam += _XS["b_vol"] * math.log(max(volume, 1.0) / vol_ref)
    if duration_days is not None:
        lam += _XS["b_dur"] * math.log(max(duration_days, 1e-3) / dur_ref)
    return EmpiricalWedge(point=lam, ci=(lam - uncertainty, lam + uncertainty), shrink=1.0)


# ---------------------------------------------------------------------------
# The comparison / test
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WedgeComparison:
    market: float
    p_lower: float
    p_upper: float
    implied: ImpliedWedge
    emp_lo: float
    emp_hi: float
    emp_point: float
    verdict: str            # "consistent" | "rich (SELL)" | "cheap (BUY)"
    wedge_gap: float        # probit-space distance outside the plausible range
    price_band: tuple[float, float]   # wedge-adjusted fair market-price band
    price_edge: float       # how far the market price sits outside that band

    @property
    def implied_mid(self) -> float:
        return self.implied.mid


def wedge_test(
    market_price: float,
    p_lower: float,
    p_upper: float,
    empirical: EmpiricalWedge,
    *,
    saturate_tol: float = 0.01,
) -> WedgeComparison:
    """Compare implied wedge (our price vs market) to the empirical wedge range.

    If our physical band saturates at 0 or 1 (e.g. a continuity-corrected barrier
    at spot), the probit transform is unbounded and the implied wedge is
    uninformative; we return verdict "n/a (p* saturated)" rather than a spurious
    extreme lambda.
    """
    iw = implied_wedge(market_price, p_lower, p_upper)
    emp_lo, emp_hi = empirical.range()

    if p_lower <= saturate_tol or p_upper >= 1.0 - saturate_tol:
        return WedgeComparison(
            market=market_price, p_lower=p_lower, p_upper=p_upper, implied=iw,
            emp_lo=emp_lo, emp_hi=emp_hi, emp_point=empirical.point,
            verdict="n/a (p* saturated)", wedge_gap=0.0,
            price_band=(p_lower, p_upper), price_edge=0.0,
        )

    # Wedge-adjusted fair market-price band (apply plausible wedge to our band).
    band_lo = wang_transform(p_lower, emp_lo)
    band_hi = wang_transform(p_upper, emp_hi)

    if iw.lo > emp_hi:          # need more positive wedge than plausible -> too rich
        verdict, gap = "rich (SELL)", iw.lo - emp_hi
        price_edge = market_price - band_hi
    elif iw.hi < emp_lo:        # need less wedge than plausible -> too cheap
        verdict, gap = "cheap (BUY)", emp_lo - iw.hi
        price_edge = band_lo - market_price
    else:
        verdict, gap, price_edge = "consistent", 0.0, 0.0

    return WedgeComparison(
        market=market_price, p_lower=p_lower, p_upper=p_upper, implied=iw,
        emp_lo=emp_lo, emp_hi=emp_hi, emp_point=empirical.point,
        verdict=verdict, wedge_gap=gap,
        price_band=(band_lo, band_hi), price_edge=max(price_edge, 0.0),
    )
