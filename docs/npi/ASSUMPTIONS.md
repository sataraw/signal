# NPI Assumption Register & Phase 0 Math Audit

Written as part of PLAN.md Phase 0 (validate the NPI implementation).
Each assumption gets one paragraph with the **direction of bias** it induces.
Findings from the math audit of `first_passage.py` / `bernoulli.py` follow,
each tied to the test that now guards it (`tests/test_ground_truth.py`).

## Assumptions

### 1. Exchangeability / iid increments (no volatility clustering)

NPI's A(n) treats the historical one-step log-returns as exchangeable, and the
lattice iterates the same one-step imprecise transition at every step — an iid
world. BTC returns show volatility clustering, so multi-step distributions are
fatter-tailed than the iid convolution of the sample. **Bias: touch and tail
probabilities biased down in high-vol regimes, up in calm regimes.** Partially
mitigated by `_regime_adjust` (Hull-White vol matching, `vol_window=30`) and by
the *block* bootstrap in `bounds.py`, which preserves local dependence.

### 2. Diffusion / no-jump (shared with the Deribit RND side)

The lattice inherits whatever jumps exist in the historical sample, so it is
less restrictive than a GBM model — but the continuity correction
(Broadie-Glasserman-Kou, `_BGK_BETA = 0.5826`) assumes a *diffusion* between
monitoring dates. Around jumps (liquidation cascades) the correction misstates
the barrier shift. **Bias: touch probabilities biased down when the path jumps
across the barrier between steps.** Same no-jump caveat as Breeden-Litzenberger
on the RND side (PLAN.md Phase 2) — note the alignment for the paper.

### 3. Constant μ and σ over the contract horizon

Increments are regime-adjusted once at pricing time and then held fixed for the
whole backward recursion; there is no intra-horizon vol dynamics. Defaults:
`vol_window=30`, `skew_aware=True`, `drift="zero"`. **Bias: for long-dated
contracts (the Dec-2026 "reach" markets) a calm current regime deflates touch
probabilities that mean-reverting vol would raise; direction follows the sign
of (long-run vol − current vol).** The `drift="zero"` default means the lattice
prices with a driftless physical measure — deliberate (stale trends shouldn't
inflate one-sided touch probabilities), but it must be documented as a choice,
and it interacts with the Phase 2 drift-adjustment work.

### 4. Physical-measure inputs

The NPI engines consume *historical* (physical-measure) returns. Polymarket
prices are also physical-measure probabilities (plus risk/liquidity premia), so
the comparison is measure-consistent — unlike the Deribit RND, which is
risk-neutral until Phase 2's adjustment. The wedge module (Wang transform)
exists precisely to model the remaining physical↔market gap. **Bias: none
inherent, but comparing raw NPI output to an unadjusted risk-neutral RND
conflates measure differences with mispricing.**

### 5. Grid discretisation (audit finding — now regression-tested)

The lattice rounds each A(n) gap to integer grid offsets (`round(a/h)`) and
places the absorbing barrier at the first grid point ≥ the true barrier. Both
effects bias probabilities **down by O(h)**, and at the default
`n_states = 121` this bias (~1.3pp in the audited configuration) *exceeds the
NPI bracket width itself* at large n, so the bracket can miss the
empirical-process truth entirely. Verified: at 481 states the bracket contains
the Monte-Carlo truth of the empirical process; convergence is monotone from
below (`test_grid_resolution_bias_direction_and_convergence`).
**Recommendation: use `n_states >= 241` for production pricing; treat 121 as a
fast preview.**

### 6. Raw NPI width is NOT a confidence interval (audit finding)

The [L, U] bracket quantifies *ambiguity within the sample* (A(n) gap
structure), which shrinks like 1/(n+1). It does **not** cover sampling noise:
at n = 4000 the bracket is ~0.004 wide while one standard deviation of
estimation error on the probability is ~0.023. Coverage of the true value comes
from the bootstrap `arbitrage_band` (measured 18/20 = 90% at the claimed
alpha=0.05 two-sided level, `test_bootstrap_band_covers_true_value`). **Never
quote the raw bracket as "our confidence interval"; always quote the band.**
Guarded by `test_raw_npi_width_ignores_sampling_noise`.

### 7. Bernoulli engine: overlapping windows inflate n

`bernoulli_crossing_probability` defaults to `stride=1` (overlapping windows),
which counts heavily autocorrelated windows as independent trials. The NPI
imprecision 1/(n+1) is then far too small — the engine looks much more certain
than it is. **Bias: overconfidence (interval too narrow), point estimate
roughly unbiased.** Use `stride=horizon` for an honest n; the docstring says
this but the *default* is the risky choice.

### 8. Time-homogeneous iteration of one-step NPI (rectangularity)

NPI is a one-step-ahead predictive inference; the lattice re-applies the same
imprecise transition for `horizon` steps and backward-inducts lower/upper
expectations. This is the standard imprecise-Markov-chain treatment (valid
under rectangularity) but it is an extension beyond what A(n) itself licenses —
the n+1-th observation updates A(n), the lattice doesn't. **Bias: understates
multi-step ambiguity slightly (imprecision does not compound with updating).**
Worth one line in the paper's method section.

## Hard-coded constants found in the code

| Constant | Value | Location | Note |
|---|---|---|---|
| `n_states` | 121 | `first_passage.py` | too coarse for production — see #5 |
| `pad_sigmas` | 4.0 | `grid_bounds` | grid roaming pad; edge-clamping beyond it |
| `_BGK_BETA` | 0.5826 | `bounds.py` | continuity correction, diffusion-only (#2) |
| `vol_window` | 30 | `bounds.py` | regime-match window; None disables |
| `drift` | `"zero"` | `bounds.py` | driftless physical measure by default (#3) |
| `alpha` | 0.05 | `arbitrage_band` | two-sided → 90% band |
| `n_boot` | 200 | `arbitrage_band` | 150 used in the live example |
| `block` | √n | `_block_bootstrap` | moving-block length |
| `friction` | 0.0 | `arbitrage_band` | Polymarket fee currently ~0 |
| `stride` | 1 | `bernoulli.py` | overlapping windows — see #7 |

## Audit verdict

The lower/upper backward recursion, A(n) mass placement (n+1 gaps at
1/(n+1) each, tails clamped to the grid), absorbing-barrier handling
(overwrite-after-expectation), and the vectorised expectation (validated
against brute force in `test_expectation_matches_bruteforce`) are all
**correct**. The two substantive issues are numerical, not conceptual:
grid-resolution bias at the default `n_states` (#5) and the interpretation
of the raw bracket (#6). Both are now regression-tested.
