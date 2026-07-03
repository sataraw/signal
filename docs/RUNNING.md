# Running Experiments and Backtests

What you need to know to run anything in this repo, and what is still missing
before full backtests are possible. Companion docs: `docs/INTERFACES.md`
(data contracts between modules) and `docs/npi/ASSUMPTIONS.md` (model
assumptions and their biases).

## Setup (5 minutes)

From the repo root on the `NPI` branch:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -e .
.venv/bin/python -m pytest tests/        # 19 tests, ~1 min — confirms the env
```

No conda, no `PYTHONPATH` hacks; run everything from the repo root.

**Mac users on Python 3.13+:** if imports mysteriously fail outside the repo
root, run `chflags -R nohidden .venv`. macOS can set the hidden flag on the
venv tree, and Python's `site.py` silently skips hidden `.pth` files, which
breaks the editable install.

## What can be experimented with today

Only the NPI side prices anything yet (the Deribit RND pipeline is still
stubs — see "What's missing" below). Entry points, all taking a plain 1-D
numpy array of historical closes:

- `npi_pricing.bounds.npi_probability(prices, spot, barrier, horizon, kind=...)`
  — "our price" as an `[L, U]` bracket; `kind="touch"` for reach/dip,
  `"terminal"` for above/below/UpDown.
- `npi_pricing.bounds.arbitrage_band(...)` — the tradeable band (bootstrap +
  frictions) with `classify(market_price) -> BUY/SELL/no-trade`.
- Live data: `src.polymarket.get_btc_contracts()` (canonical Polymarket
  fetcher), `src.market_data.binance_closes` / `binance_spot` /
  `binance_price_at` (the last one recovers Up/Down window-start strikes).
- `examples/polymarket_btc.py` and `examples/wedge_test.py` are working
  end-to-end templates for exactly this loop; `examples/demo.py` is the
  offline synthetic version.

**Experiment knobs** — the assumption toggles, all keyword arguments to the
two functions above:

| knob | values | what it varies |
|---|---|---|
| `drift` | `"zero"` / `"recent"` / `"full"` | which historical trend the lattice trusts |
| `vol_window` | int or None | Hull-White regime matching window (None disables) |
| `skew_aware` | bool | separate up/down tail rescaling |
| `continuous_monitoring` | bool | Broadie-Glasserman-Kou barrier correction (touch only) |
| `friction` | float | half-spread + fees, in probability terms |
| `alpha`, `n_boot` | float, int | band confidence level and bootstrap size |

What each assumes and which way it biases results: `docs/npi/ASSUMPTIONS.md`.

## Three conventions that prevent silently wrong results

1. **Horizon counts steps of the history interval.** Daily closes → horizon in
   days; minute closes → horizon in minutes. Mixing them fails silently with
   plausible-looking numbers. (`docs/INTERFACES.md` §2.)
2. **The raw NPI `[L, U]` is not a confidence interval.** It quantifies
   ambiguity within the sample and gets *narrower* with more data while
   ignoring sampling noise. Any experiment quoting uncertainty must use
   `arbitrage_band`, which is calibrated (measured 90% coverage at the claimed
   level — `tests/test_ground_truth.py`).
3. **Use `n_states=241` or higher.** The default 121 has a known downward bias
   larger than the bracket width itself (`docs/npi/ASSUMPTIONS.md` §5).

## What's still missing before real backtests

1. **The RND pricer does not exist yet.** `src/rnd.py`, `src/blackscholes.py`,
   `src/contracts.py`, `src/bands.py` are stubs (milestones M3–M6). Until they
   are built, "experiments" means NPI-vs-market only, not NPI-vs-RND-vs-market.
2. **No historical matched contracts.** The fetcher returns *live* markets. A
   backtest needs resolved contracts with known outcomes plus the BTC history
   at their open (contract matching + Binance history). Until then, backtests
   can only be smoke-tested on synthetic contracts.
3. **Metrics must be frozen before tuning** (the team's overfitting rule).
   Brier score, log loss, calibration curves, interval coverage/width,
   band-violation PnL, and the train/test split need sign-off *before* anyone
   starts turning the knobs above — otherwise the knob-turning is the
   overfitting.
4. **Reproducibility discipline:** live fetches are not repeatable. Every
   experiment should run off saved snapshots (`valid_btc_markets_*.json`,
   chain snapshots via `src/storage.py`) with the snapshot filename recorded
   next to the results.
5. **Blocked externally:** the minute-level data pipeline repo URL/branch —
   needed for the 30-min resolution study and Up/Down backtests.
