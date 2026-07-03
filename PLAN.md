# Working Plan — Quinten's Tasks (NPI Pricing & Pipeline Integration)

Structure for working through Quinten's to-dos from the team meeting. Phases are
ordered by dependency; each maps to the numbered tasks from the meeting summary
and ends with explicit verification criteria.

**Status of open decisions** (from planning discussion):
- Wang transform ownership: **undecided** — framework is designed so it plugs in either way.
- Drift adjustment lives on the **Deribit RND side** (risk-neutral → physical).
- Minute-level BTC data comes from the **existing data pipeline on GitHub** (URL/branch TBD).
- Restructuring the NPI project and the pipeline to work together is an explicit task.

---

## Phase 0 — Validate the NPI implementation *(tasks 1, 2)*

1. Run the existing 11 tests and the three examples; fix anything broken.
2. Math audit of `first_passage.py` and `bernoulli.py`: A(n) mass placement,
   lower/upper backward recursion, absorbing-barrier handling, grid-resolution
   sensitivity.
3. Add ground-truth tests:
   - GBM paths with known analytic one-touch probability → assert the NPI
     `[L, U]` brackets it.
   - Calibration/coverage test: true outcomes fall inside the interval at the
     claimed rate.
   - Convergence test as `n` grows.
4. Enumerate and document the NPI-side assumptions explicitly:
   - **Diffusion/no-jump**: the lattice assumes continuous sample paths (GBM);
     BTC has jump risk — document the expected direction of bias (jumps widen
     the true interval; NPI may undercover during high-vol regimes).
   - **Constant μ and σ over the lattice horizon**: stationarity assumption;
     note sensitivity to window choice for parameter estimation.
   - **Grid-resolution discretisation error**: document convergence rate and the
     minimum `n` where the error is negligible relative to the interval width.
   - **Physical-measure inputs**: NPI takes a drift μ estimated from historical
     data; document that this is a physical-measure quantity, distinct from the
     risk-neutral drift used in the Deribit RND (relevant to Phase 2 alignment).

**Verify:** all tests pass; intervals bracket analytic values on synthetic data;
written assumption register (one paragraph per assumption with direction of bias)
checked into the repo alongside the phase.

---

## Phase 1 — Restructure so NPI and the pipeline work together *(task 3)*

1. Convert `NPI Pricing/` from a standalone folder (space in name, own `.venv`,
   `PYTHONPATH` hack) into a proper package alongside `src/` with one shared
   environment and one requirements file.
2. Single data layer: NPI consumes BTC history from the same source/schema as
   the rest of the pipeline — including the GitHub data pipeline that provides
   minute-level data. **Needed: that repo URL/branch.**
3. Define and document the interfaces: contract spec in (aligned with Awar's
   `PolymarketContract` dataclass), price history in, quote interval out.
   This doc doubles as the schema/scope contribution due Tuesday.

**Verify:** `pytest` runs everything from repo root in one env; one example
prices a live Polymarket contract through the shared data layer end-to-end.

---

## Phase 2 — Drift adjustment on the Deribit RND side *(task 5)*

0. **Enumerate and document the RND-side assumptions before touching the code.**
   These are distinct from the drift assumption and most are not fixed by the
   measure change — they need to be recorded so the paper can address them:
   - **Log-normal tail extrapolation**: strikes beyond the traded range are
     extrapolated assuming a log-normal shape. BTC tails are heavier; this
     biases above-ATM digital prices downward and below-ATM upward. Document the
     sensitivity: how much does the "above $X" price change when the tail
     exponent is varied ±20%?
   - **Cubic spline smoothness (no-kink assumption)**: the IV smile is assumed
     smooth across all strikes. Kinks or vol surface discontinuities (e.g. around
     round strikes) are smoothed away. Flag if the spline oscillates between
     sparse strikes (see CLAUDE.md §16).
   - **Diffusion / no-jump**: Breeden-Litzenberger requires a diffusion process.
     BTC jumps (e.g. liquidation cascades) cause the RND to understate tail mass
     precisely when it matters most. This assumption is *shared* with the NPI
     lattice (Phase 0) — note the alignment.
   - **Vol risk premium in Deribit IVs**: Deribit implied vols embed a variance
     risk premium, so the "risk-neutral" density already leans toward
     overweighting tails relative to a pure risk-neutral measure. The drift
     adjustment in step 2 below partially counteracts this but does not eliminate
     it — document what remains.
   - **Expiry / settlement basis**: the nearest Deribit expiry may not match the
     Polymarket contract's settlement date. Document the mismatch in days and the
     resulting fair-value uncertainty (roughly `vega × IV × sqrt(Δt/T)`).
   - **European-style settlement**: Deribit options settle European; Polymarket
     "above $X" contracts resolve on a 30-min TWAP (addressed in Phase 3).
     Record here that Phase 3 covers this so the assumption register is complete.

1. Implement a drift estimator (rolling-window mean log-return as baseline;
   document window choice and alternatives).
2. Apply a measure change to the Breeden-Litzenberger RND in `src/rnd.py`:
   shift the risk-neutral density toward a physical one using estimated μ
   (simplest: exponential tilting / mean shift; document the assumption).
3. Expose zero-drift vs estimated-drift as a toggle so the backtest can measure
   the effect — one of the experiments the team listed.
4. For each assumption in step 0, add a sensitivity flag or toggle where
   feasible (e.g. tail-exponent parameter, spline type switch) so Phase 4 can
   run experiments varying them independently.

**Verify:** adjusted RND still integrates to 1; digital prices move in the
expected direction for μ > r; unit tests on synthetic log-normal cases with
closed-form answers; assumption register doc updated with Phase 2 items.

---

## Phase 3 — 30-min resolution adjustment *(task 7)*

1. Analytical: under a diffusion, derive the distribution of the 30-min average
   vs the endpoint and the resulting digital-price difference (largest near the
   strike, shrinks with horizon).
2. Empirical: replay both resolution rules on minute-level history from the data
   pipeline — how often do outcomes disagree, and what is the fair-value gap as
   a function of moneyness and horizon?
3. Package as an adjustment function the pricers can apply — this is the
   distinct methodological contribution, so it gets its own writeup.

**Verify:** analytical and simulated results agree; empirical disagreement rates
reported per contract horizon (daily / 4h / 1h).

---

## Phase 4 — Comparison framework and metrics *(tasks 6, 9)*

1. Pluggable pricer interface: every method (NPI lattice, NPI Bernoulli,
   Deribit RND digital, drift-adjusted RND, naive baselines) implements
   `price(contract, market_state) → point or interval`.
   The **Wang transform gets a slot in this interface** — ownership undecided,
   so the plug is designed for either owner to fill without rework.
   Each pricer implementation must also carry a short `assumptions` metadata
   dict (keys: `measure`, `tail_model`, `jump_model`, `settlement`) so the
   framework can surface which assumption set produced each result.
2. Metrics defined **before** experiments run: Brier score, log loss,
   calibration curves, interval coverage + width for NPI, and band-violation
   PnL as the economic metric.
3. Specify the train/test split: development window vs untouched out-of-sample
   window, per the meeting's overfitting rule.
4. Treat **one-touch and terminal ("above $X on date") contracts as separate
   contract types** from the start — the lattice already supports both
   (absorbing vs non-absorbing recursion).
5. **Assumption sensitivity experiments** — run the following as a defined
   experiment batch (design before backtest runs, per the overfitting rule):
   - Log-normal vs flat (constant-IV) tail extrapolation.
   - Spline (`CubicSpline`) vs monotone (`PchipInterpolator`) IV fitting.
   - Zero-drift vs rolling-window-drift RND (already toggled in Phase 2).
   - With vs without 30-min resolution adjustment (Phase 3 toggle).
   Results reported as Δ(Brier) per assumption flip so the paper can quantify
   each assumption's materiality.

**Verify:** a metrics spec document the team signs off on before any tuning;
assumption sensitivity experiment design included in that sign-off.

---

## Phase 5 — Backtest harness *(task 4)*

1. Loop: for each matched contract (Awar's matching) × each pricer → price,
   compare to Polymarket, resolve outcome, score.
2. Runs on the shared schemas from Phase 1; smoke-testable on synthetic
   contracts before real matched data lands.

**Verify:** end-to-end run on ≥1 real daily contract producing a results table
with all pricers and metrics.

---

## Task 8 — Scope / research proposal

The outputs of Phases 2–4 (drift spec, resolution writeup, metrics spec) are
exactly the sections owed to the scope document — drafted as the phases
complete so Tuesday isn't a scramble.

---

## Dependencies on teammates

| Needed | From | Blocks |
|---|---|---|
| Data-pipeline repo URL/branch (minute-level BTC) | team | Phase 1.2, Phase 3.2 |
| Latest contract-matching schema | Awar | Phase 1.3 |
| Matched contracts (historical) | Awar | Phase 5 |
| Historical Bitcoin data (Binance) | Max-Loic | Phase 5 |

Phases 0, 2, 3 (analytical part) and most of 4 have **no external blockers**.

## Open items to raise with the team

- Up/Down markets on `polymarket_data_pull` have strikes that can't be fetched
  from the API (workaround via Chainlink data streams has a 1-minute delay) —
  should become a GitHub issue with a matching rule decided by Awar.
- Wang transform ownership.
- Exact meaning/placement of the drift adjustment (confirmed here as RND-side,
  but should be aligned with Yigit's drift/skew investigation).
