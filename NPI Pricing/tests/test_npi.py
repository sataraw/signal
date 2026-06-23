"""Sanity + correctness tests. Run:  python -m pytest  (or python tests/test_npi.py)."""

from __future__ import annotations

import numpy as np

from npi_pricing.bernoulli import npi_bernoulli, bernoulli_crossing_probability
from npi_pricing.first_passage import ImpreciseLattice, first_passage_probability
from npi_pricing.data import make_sample_prices
from npi_pricing.pricer import price_one_touch


def test_bernoulli_formula():
    r = npi_bernoulli(successes=3, trials=9)
    assert np.isclose(r.lower, 3 / 10)
    assert np.isclose(r.upper, 4 / 10)
    # edges
    assert np.isclose(npi_bernoulli(0, 0).lower, 0.0)
    assert np.isclose(npi_bernoulli(0, 0).upper, 1.0)


def _brute_expectation(lattice: ImpreciseLattice, f: np.ndarray, lower: bool) -> np.ndarray:
    """Reference O(S * gaps * width) implementation to validate the vectorised one."""
    s = lattice.n_states
    out = np.zeros(s)
    for x in range(s):
        acc = 0.0
        for (o_lo, o_hi), w in zip(lattice._offsets, lattice._weights):
            # mirror the fast path: off-grid windows clamp to the nearest edge
            lo = int(np.clip(x + o_lo, 0, s - 1))
            hi = int(np.clip(x + o_hi, 0, s - 1))
            seg = f[lo : hi + 1]
            acc += w * (seg.min() if lower else seg.max())
        out[x] = acc
    return out


def test_expectation_matches_bruteforce():
    rng = np.random.default_rng(0)
    increments = rng.normal(0, 0.01, size=60)
    lattice = ImpreciseLattice(increments, log_lo=-0.3, log_hi=0.3, n_states=41)
    f = rng.random(lattice.n_states)
    for lower in (True, False):
        fast = lattice._expectation(f, lower=lower)
        slow = _brute_expectation(lattice, f, lower)
        assert np.allclose(fast, slow), f"mismatch (lower={lower})"


def test_first_passage_bounds_and_monotonicity():
    prices = make_sample_prices(n_days=500, seed=1)
    spot = float(prices[-1])
    barrier = spot * 1.03

    res = first_passage_probability(prices, spot, barrier, horizon=15)
    assert 0.0 <= res.lower <= res.upper <= 1.0

    # more time -> (weakly) higher touch probability
    short = first_passage_probability(prices, spot, barrier, horizon=5)
    longer = first_passage_probability(prices, spot, barrier, horizon=30)
    assert longer.lower >= short.lower - 1e-9
    assert longer.upper >= short.upper - 1e-9

    # nearer barrier -> higher touch probability
    near = first_passage_probability(prices, spot, spot * 1.01, horizon=15)
    far = first_passage_probability(prices, spot, spot * 1.08, horizon=15)
    assert near.upper >= far.upper - 1e-9


def test_already_touched_is_certain():
    prices = make_sample_prices(n_days=300, seed=2)
    spot = float(prices[-1])
    # barrier at/just below spot for an up-touch is already crossed
    res = first_passage_probability(prices, spot, spot, horizon=10, up=True)
    assert np.isclose(res.lower, 1.0) and np.isclose(res.upper, 1.0)


def test_terminal_le_touch():
    # Being past the level *at expiry* is never more likely than touching it
    # at *any* time before expiry (one-touch dominates terminal).
    from npi_pricing.first_passage import ImpreciseLattice, grid_bounds
    prices = make_sample_prices(n_days=500, seed=5)
    spot = float(prices[-1])
    barrier = spot * 1.04
    incr = np.diff(np.log(prices))
    lo, hi = grid_bounds(incr, np.log(spot), np.log(barrier), 20)
    lat = ImpreciseLattice(incr, lo, hi, 161)
    touch = lat.first_passage(np.log(spot), np.log(barrier), 20, up=True)
    term = lat.terminal(np.log(spot), np.log(barrier), 20, up=True)
    assert term.lower <= touch.lower + 1e-9
    assert term.upper <= touch.upper + 1e-9
    assert 0.0 <= term.lower <= term.upper <= 1.0


def test_arbitrage_band_basic():
    from npi_pricing.bounds import arbitrage_band
    prices = make_sample_prices(n_days=600, seed=6)
    spot = float(prices[-1])
    band = arbitrage_band(prices, spot, spot * 1.05, horizon=20,
                          up=True, kind="touch", n_boot=40, n_states=81)
    assert 0.0 <= band.arb_lower <= band.arb_upper <= 1.0
    assert band.arb_lower <= band.npi_lower + 1e-9  # band brackets the point est.
    assert band.arb_upper >= band.npi_upper - 1e-9
    assert "BUY YES" in band.classify(band.arb_lower - 0.05)[0]
    assert "SELL YES" in band.classify(band.arb_upper + 0.05)[0]


def test_regime_adjust_drift_and_skew():
    from npi_pricing.bounds import _regime_adjust, _semi_deviations
    # full history: symmetric large moves; recent window: calm UPSIDE, normal downside
    base = np.array([0.10, -0.10] * 60)        # full up-vol == down-vol
    recent = np.array([0.005, -0.10] * 15)     # recent up-vol << down-vol
    hist = np.concatenate([base, recent])      # last 30 obs are the recent window

    # (1) drift reset to the chosen regime
    out = _regime_adjust(hist, vol_window=30, skew_aware=True, drift="recent")
    assert abs(out.mean() - recent.mean()) < 1e-9
    assert abs(_regime_adjust(hist, vol_window=30, drift="zero").mean()) < 1e-9

    # (2) skew-aware shrinks the up-tail far more than symmetric scaling does
    sym = _regime_adjust(hist, vol_window=30, skew_aware=False, drift="recent")
    su_skew, sd_skew = _semi_deviations(out - out.mean())
    su_sym, sd_sym = _semi_deviations(sym - sym.mean())
    assert su_skew < su_sym       # upside scaled down on its own
    assert sd_skew > sd_sym       # downside not dragged down with it

    # (3) None disables, leaving increments untouched
    assert np.allclose(_regime_adjust(hist, None), hist)


def test_norm_roundtrip_and_wang():
    from npi_pricing.wedge import norm_cdf, norm_ppf, wang_transform, implied_wedge
    for p in (0.01, 0.1, 0.27, 0.5, 0.73, 0.9, 0.99):
        assert abs(norm_cdf(norm_ppf(p)) - p) < 1e-6
    # Wang transform and implied wedge are exact inverses
    p_star, lam = 0.30, 0.176
    m = wang_transform(p_star, lam)
    iw = implied_wedge(m, p_star, p_star)  # degenerate band -> point
    assert abs(iw.mid - lam) < 1e-6
    # positive wedge => market above physical (overpricing); FLB direction
    assert m > p_star


def test_wedge_verdicts():
    from npi_pricing.wedge import EmpiricalWedge, wedge_test, wang_transform
    emp = EmpiricalWedge(point=0.176, ci=(0.12, 0.23))
    # market exactly at physical + mid empirical wedge -> consistent
    p_lo, p_hi = 0.18, 0.22
    fair = wang_transform(0.20, 0.176)
    assert wedge_test(fair, p_lo, p_hi, emp).verdict == "consistent"
    # market far above what any plausible wedge gives -> rich (SELL)
    rich = wang_transform(0.22, 0.60)
    assert wedge_test(rich, p_lo, p_hi, emp).verdict.startswith("rich")
    # market below physical (negative implied wedge) -> cheap (BUY)
    cheap = wang_transform(0.18, -0.30)
    assert wedge_test(cheap, p_lo, p_hi, emp).verdict.startswith("cheap")


def test_quote_and_decision():
    prices = make_sample_prices(n_days=600, seed=3)
    spot = float(prices[-1])
    q = price_one_touch(prices, spot, spot * 1.05, horizon=20, payout=1.0)
    assert q.price_lower <= q.price_upper
    assert "BUY" in q.decision(q.price_lower - 0.05)
    assert "SELL" in q.decision(q.price_upper + 0.05)
    assert "NO TRADE" in q.decision(q.price_mid)


def test_bernoulli_crossing_runs():
    prices = make_sample_prices(n_days=400, seed=4)
    spot = float(prices[-1])
    r = bernoulli_crossing_probability(prices, spot, spot * 1.02, horizon=10)
    assert 0.0 <= r.lower <= r.upper <= 1.0
    assert np.isclose(r.imprecision, 1.0 / (r.n + 1))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
