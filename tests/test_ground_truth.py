"""Ground-truth tests against analytic GBM values and the empirical process.

Phase 0 validation (PLAN.md). Three layers, matching what the NPI bracket can
and cannot claim:

1. *Numerics* — the lattice [L, U] must bracket the touch/terminal probability
   of the EMPIRICAL process (iid draws from the historical increments), because
   that is the process NPI actually reasons about. Ground truth via Monte Carlo
   resampling from the same sample.
2. *Convergence* — imprecision shrinks as n grows and the midpoint approaches
   the true-parameter GBM value.
3. *Calibration* — the raw NPI width does NOT account for sampling noise (a
   documented audit finding); it is the bootstrap arbitrage band that claims
   ~90% coverage of the true value, so coverage is asserted on the band.

Run:  python -m pytest tests/test_ground_truth.py
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from npi_pricing.bounds import arbitrage_band
from npi_pricing.first_passage import ImpreciseLattice, grid_bounds
from npi_pricing.wedge import norm_cdf

# One fixed GBM world shared by the numerics tests.
M_STEP, S_STEP = 0.0005, 0.02   # per-step log-return mean / sd
B_LOG = 0.03                    # log-distance from spot to the up-barrier
HORIZON = 15
N_INCR = 4000
SEED = 42


@lru_cache(maxsize=None)
def _world():
    rng = np.random.default_rng(SEED)
    incr = rng.normal(M_STEP, S_STEP, N_INCR)
    spot_log = 0.0
    barrier_log = B_LOG
    return incr, spot_log, barrier_log, rng


@lru_cache(maxsize=None)
def _lattice_results(n_states: int):
    incr, spot_log, barrier_log, _ = _world()
    lo, hi = grid_bounds(incr, spot_log, barrier_log, HORIZON)
    lat = ImpreciseLattice(incr, lo, hi, n_states)
    term = lat.terminal(spot_log, barrier_log, HORIZON, up=True)
    touch = lat.first_passage(spot_log, barrier_log, HORIZON, up=True)
    return term, touch


@lru_cache(maxsize=None)
def _empirical_mc():
    """Terminal/touch probability of the empirical process by resampling MC."""
    incr, _, barrier_log, rng = _world()
    z = rng.choice(incr, size=(400_000, HORIZON), replace=True)
    cum = np.cumsum(z, axis=1)
    p_term = float((cum[:, -1] >= barrier_log).mean())
    p_touch = float((cum.max(axis=1) >= barrier_log).mean())
    return p_term, p_touch


def test_terminal_brackets_empirical_process():
    term, _ = _lattice_results(481)
    p_term, _ = _empirical_mc()
    tol = 0.0075  # MC error + O(h) grid bias at 481 states
    assert term.lower - tol <= p_term <= term.upper + tol
    assert term.upper - term.lower < 0.05  # bracket is informative, not vacuous


def test_touch_brackets_empirical_process():
    _, touch = _lattice_results(481)
    _, p_touch = _empirical_mc()
    tol = 0.0075
    assert touch.lower - tol <= p_touch <= touch.upper + tol
    assert touch.upper - touch.lower < 0.05


def test_grid_resolution_bias_direction_and_convergence():
    # Audit finding: coarse grids bias BOTH probabilities downward (offset
    # rounding + effective barrier sitting up to one grid step above the true
    # barrier). Refining the grid must move the bracket up and stabilise.
    coarse_t, coarse_fp = _lattice_results(121)
    mid_t, mid_fp = _lattice_results(241)
    fine_t, fine_fp = _lattice_results(481)
    assert coarse_t.midpoint < fine_t.midpoint
    assert coarse_fp.midpoint < fine_fp.midpoint
    assert abs(fine_t.midpoint - mid_t.midpoint) < 0.01
    assert abs(fine_fp.midpoint - mid_fp.midpoint) < 0.01


def test_convergence_in_n():
    # Imprecision shrinks with sample size; midpoint approaches the
    # true-parameter GBM terminal probability.
    p_true = norm_cdf((HORIZON * M_STEP - B_LOG) / (S_STEP * np.sqrt(HORIZON)))
    widths, errs = [], []
    for n in (250, 1000, 4000):
        rng = np.random.default_rng(SEED)
        incr = rng.normal(M_STEP, S_STEP, n)
        lo, hi = grid_bounds(incr, 0.0, B_LOG, HORIZON)
        lat = ImpreciseLattice(incr, lo, hi, 241)
        term = lat.terminal(0.0, B_LOG, HORIZON, up=True)
        widths.append(term.imprecision)
        errs.append(abs(term.midpoint - p_true))
    assert widths[0] > widths[1] > widths[2]
    assert errs[2] < 0.05


def test_raw_npi_width_ignores_sampling_noise():
    # Documented audit finding: at large n the NPI bracket is far narrower than
    # the sampling spread of the estimate, so it must NOT be read as a
    # confidence interval for the true value. Guard the fact so nobody
    # "simplifies away" the bootstrap band later.
    term, _ = _lattice_results(481)
    sd_mean_effect = S_STEP / np.sqrt(N_INCR) * HORIZON  # sd of total-drift error
    p_sensitivity = sd_mean_effect / (S_STEP * np.sqrt(HORIZON))  # in z-units
    assert term.imprecision < p_sensitivity  # bracket narrower than 1 sd of noise


def test_bootstrap_band_covers_true_value():
    # Calibration: the arbitrage band (alpha=0.05, two-sided ~90%) should cover
    # the true-parameter terminal probability at roughly the claimed rate.
    # Pre-verified: 18/20 coverage with these seeds; assert a robust floor.
    p_true = norm_cdf((HORIZON * M_STEP - B_LOG) / (S_STEP * np.sqrt(HORIZON)))
    covered = 0
    n_worlds = 20
    for k in range(n_worlds):
        rng = np.random.default_rng(1000 + k)
        incr = rng.normal(M_STEP, S_STEP, 750)
        prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(incr)]))
        spot = float(prices[-1])
        barrier = spot * np.exp(B_LOG)
        band = arbitrage_band(
            prices, spot, barrier, HORIZON, up=True, kind="terminal",
            n_states=241, n_boot=60, vol_window=None, seed=k,
        )
        covered += band.arb_lower <= p_true <= band.arb_upper
    assert covered >= 15  # >= 75% observed floor for a 90% claimed band


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
