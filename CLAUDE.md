# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

# Implementation Plan: Polymarket BTC Pricing Pipeline

## 0. Orientation — What You Are Building and Why

Before writing a single line of code, understand the conceptual chain:

1. Deribit is a crypto options exchange where traders buy and sell BTC options at many different strike prices and expiries. The prices of these options collectively encode a *probability distribution* over where BTC will be at expiry. This distribution is called the **risk-neutral density (RND)**.

2. Polymarket is a prediction market where you can bet on things like "will BTC be above $100k on July 31?" These contracts pay $1 if true, $0 if false — their price is therefore a probability between 0 and 1.

3. The core insight of this project: a Polymarket "Above $100k" contract is *mathematically identical* to a cash-or-nothing binary call option. We can price it by integrating the Deribit RND above $100k. If Polymarket's price differs from that integral by more than a friction-based band, that is a candidate arbitrage.

Your job is to build the software infrastructure that makes this computation automatic and repeatable. You are not designing the theory — your team owns that. You are building the data and computation engine that produces numbers for them to analyse.

---

## 1. Repository Structure

Create the repository exactly as follows. Do not deviate from this layout — later modules will depend on it. You will be working inside of an anaconda virtual environment that has been created already. You can use it with `conda activate signal`. 

```
polymarket-btc-pricer/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── config/
│   └── settings.py              # All constants and config in one place
│
├── data/
│   ├── fetchers/
│   │   ├── __init__.py
│   │   ├── deribit.py           # All Deribit API calls
│   │   └── polymarket.py        # All Polymarket API calls
│   ├── storage/
│   │   ├── __init__.py
│   │   └── local.py             # Save/load snapshots to parquet
│   └── schemas.py               # Dataclass/TypedDict definitions
│
├── pricing/
│   ├── __init__.py
│   ├── black_scholes.py         # BS formula and Greeks
│   ├── rnd.py                   # IV smile fitting + Breeden-Litzenberger
│   └── contracts.py             # Contract pricing (Above, Range)
│
├── analysis/
│   ├── __init__.py
│   └── band.py                  # No-arb band construction + violation detection
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_rnd_validation.ipynb
│   └── 03_contract_pricing.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_black_scholes.py
│   ├── test_rnd.py
│   └── test_contracts.py
│
└── scripts/
    └── run_pipeline.py          # Top-level entry point
```

`.gitignore` must include: `data/snapshots/`, `.env`, `__pycache__/`, `.ipynb_checkpoints/`, `*.pyc`

`.env.example`:
```
# Copy to .env and fill in. Never commit .env.
DERIBIT_BASE_URL=https://www.deribit.com/api/v2/public
POLYMARKET_BASE_URL=https://clob.polymarket.com
RISK_FREE_RATE=0.05
```

---

## 2. `requirements.txt`

```
requests==2.31.0
pandas==2.2.0
numpy==1.26.0
scipy==1.12.0
pyarrow==15.0.0      # parquet support
python-dotenv==1.0.0
matplotlib==3.8.0
plotly==5.18.0
pytest==8.0.0
jupyter==1.0.0
```

No exotic libraries. Everything here is standard scientific Python. Install with `pip install -r requirements.txt`.

---

## 3. `config/settings.py`

This file is the single source of truth for all constants. Never hardcode numbers elsewhere in the codebase.

```python
import os
from dotenv import load_dotenv

load_dotenv()

# API
DERIBIT_BASE_URL = os.getenv("DERIBIT_BASE_URL", "https://www.deribit.com/api/v2/public")
POLYMARKET_BASE_URL = os.getenv("POLYMARKET_BASE_URL", "https://clob.polymarket.com")

# Market
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", 0.05))
CURRENCY = "BTC"

# RND extraction
MIN_STRIKES = 8          # Minimum number of strikes needed for a valid smile fit
TAIL_PERCENTILE = 0.01   # Log-normal tail extrapolation cutoff
STRIKE_GRID_POINTS = 2000  # Resolution of the interpolated strike grid

# Storage
SNAPSHOT_DIR = "data/snapshots"

# No-arb band
POLYMARKET_FEE = 0.02    # 2% taker fee
GAS_COST_USD = 1.0       # Approximate Polygon gas cost per transaction
```

---

## 4. `data/schemas.py`

Define explicit data structures for everything that flows between modules. This prevents silent type errors and makes the pipeline easy to reason about.

## 5. Module: `data/fetchers/deribit.py`

This module owns all communication with the Deribit API. Nothing outside this module should call `requests` for Deribit data.

**Functions to implement:**

### `get_instruments() -> pd.DataFrame`

Fetches all active BTC option instruments.

- Endpoint: `GET /get_instruments?currency=BTC&kind=option&expired=false`
- Returns a DataFrame with columns: `instrument_name`, `strike`, `expiry_timestamp`, `option_type`
- Parse `option_type` from the instrument name suffix (`-C` = call, `-P` = put)
- Parse `strike` from the instrument name (e.g. `BTC-31JUL25-100000-C` → `100000`)

### `get_order_book(instrument_name: str) -> dict`

Fetches the order book for a single instrument.

- Endpoint: `GET /get_order_book?instrument_name={name}&depth=1`
- Returns the raw result dict. You need: `bid_iv`, `ask_iv`, `best_bid_price`, `best_ask_price`, `underlying_price`, `greeks.delta`, `open_interest`
- Return `None` if the request fails or the result has no bids/asks

### `get_option_chain(expiry_timestamp: int) -> OptionChain`

The main function you will use from outside this module.

- Calls `get_instruments()`, filters to the target expiry
- Calls `get_order_book()` for every instrument at that expiry
- Assembles into an `OptionChain` dataclass
- **Cleaning rules to apply:**
  - Drop instruments where `bid_iv` or `ask_iv` is 0 or null (no market)
  - Drop instruments where `open_interest` == 0
  - Compute `mid_iv = (bid_iv + ask_iv) / 2`
  - Compute `mid_price = (best_bid_price + best_ask_price) / 2` (in USD: multiply by `underlying_price`)
  - Sort by strike ascending
- Log a warning if fewer than `MIN_STRIKES` strikes survive cleaning

### `find_nearest_expiry(target_timestamp: int) -> int`

Given a target timestamp (e.g. from a Polymarket contract), returns the Deribit expiry timestamp closest to it.

- Use `get_instruments()` to list available expiries
- Return the one minimising `abs(expiry - target_timestamp)`
- Log the date mismatch in days if it exceeds 3 days (this affects settlement basis)

**Error handling:** wrap all `requests.get()` calls in try/except. On network error, log and re-raise. On API error response, log the error message and return `None`. Never silently swallow exceptions.

---

## 6. Module: `data/fetchers/polymarket.py`

This module fetches Polymarket contract data. Keep it isolated — the API may change and you don't want Polymarket-specific parsing scattered through the codebase.

**Functions to implement:**

### `get_btc_contracts() -> list[PolymarketContract]`

Fetches active BTC price prediction markets from Polymarket's CLOB API.

- Endpoint: `GET https://clob.polymarket.com/markets` with filtering
- You are looking for markets whose question contains "BTC" or "Bitcoin" and are price-related ("above", "reach", "range", "dip")
- For each matching market, parse into a `PolymarketContract`
- The `polymarket_price` should be the current mid-price of the YES token (bid + ask / 2)
- Classify `bet_type` by parsing the question text: "above" / "reach" / "below" map to `"above"`, `"range"` maps to `"range"`, "dip" / "drop" map to `"dip"`

**Important:** Polymarket's public API is not heavily documented. You will need to inspect the response carefully. Start by hitting the endpoint manually in your browser or with `curl` and printing the raw response before writing parsing logic. The contract question text is your primary source of truth for `bet_type` and strike values.

### `parse_strike_from_question(question: str) -> tuple[Optional[float], Optional[float]]`

Parses strike price(s) from a question string like "Will BTC be above $100,000 on July 31?".

- Use regex to extract dollar amounts: `r'\$[\d,]+'`
- Strip commas, convert to float
- For Above/Reach/Dip: return `(strike, None)`
- For Range: return `(lower_strike, upper_strike)`
- Return `(None, None)` if parsing fails and log a warning — do not crash

---

## 7. Module: `data/storage/local.py`

You will be calling the Deribit API frequently during development. Cache everything locally to avoid rate limits and enable reproducible research.

**Functions to implement:**

### `save_option_chain(chain: OptionChain, label: str = "")`

- Saves `chain.quotes` as parquet to `data/snapshots/chains/`
- Filename: `btc_chain_{expiry_date}_{timestamp}{label}.parquet`
- Also saves metadata (underlying price, fetch time, expiry) as a sidecar JSON

### `load_option_chain(filepath: str) -> OptionChain`

- Loads parquet + sidecar JSON, reconstructs `OptionChain`

### `save_rnd(rnd: RNDResult, label: str = "")`

- Saves strikes and density as parquet to `data/snapshots/rnds/`

### `load_rnd(filepath: str) -> RNDResult`

### `list_snapshots(kind: str) -> list[str]`

- `kind` is `"chains"` or `"rnds"`
- Returns sorted list of available snapshot filepaths

**Rule:** all paths are relative to the project root. Create directories if they don't exist. Never hardcode absolute paths.

---

## 8. Module: `pricing/black_scholes.py`

Implement Black-Scholes from scratch. Do not use a library for this — the formula is 10 lines and you need fine-grained control over it.

**Functions to implement:**

### `call_price(S, K, T, r, sigma) -> float`

Standard European call price. All inputs are floats:
- `S`: current underlying price
- `K`: strike
- `T`: time to expiry in years
- `r`: continuously compounded risk-free rate
- `sigma`: implied volatility (annualised, as a decimal e.g. 0.8 for 80%)

Formula:
```
d1 = (ln(S/K) + (r + 0.5·σ²)·T) / (σ·√T)
d2 = d1 - σ·√T
C  = S·N(d1) - K·e^(-rT)·N(d2)
```
where `N()` is the standard normal CDF (`scipy.stats.norm.cdf`).

### `put_price(S, K, T, r, sigma) -> float`

Same structure. Use put-call parity: `P = C - S + K·e^(-rT)`.

### `implied_vol(market_price, S, K, T, r, option_type="call") -> Optional[float]`

Invert Black-Scholes to find the IV that matches a given market price.

- Use `scipy.optimize.brentq` on the interval `[1e-6, 10.0]`
- Return `None` if the solver fails to converge or the price is below intrinsic value
- This will be used as a cross-check, not in the main pipeline

### `vectorized_call_price(S, K_array, T, r, sigma_array) -> np.ndarray`

Vectorised version operating on arrays of `K` and `sigma`. Use `numpy` operations throughout — no Python loops. This is the performance-critical path.

**Test requirement:** Write tests in `tests/test_black_scholes.py` that verify:
- ATM call price equals the Brenner-Subrahmanyam approximation: `C ≈ 0.4·S·σ·√T` to within 2%
- Put-call parity holds to machine precision
- `implied_vol(call_price(S, K, T, r, σ), ...) ≈ σ` round-trips correctly

---

## 9. Module: `pricing/rnd.py`

This is the core scientific module. Take extra care here.

**Conceptual reminder:** Breeden-Litzenberger states that the risk-neutral density is the second derivative of the call price function with respect to strike, scaled by `e^(rT)`. You cannot take this derivative on raw noisy discrete prices. You must first fit a smooth curve through the IV smile, convert back to prices, then differentiate.

**Functions to implement:**

### `fit_iv_smile(strikes: np.ndarray, mid_ivs: np.ndarray) -> CubicSpline`

Fits a smooth cubic spline through the observed (strike, IV) points.

- Input: arrays of strikes and mid IVs from the cleaned option chain (calls only)
- Use `scipy.interpolate.CubicSpline` with `bc_type='not-a-knot'`
- Return the fitted spline object (callable)
- Validate: the spline must be monotonically reasonable — flag if any interpolated IV drops below 0.01 (1%)

### `extrapolate_tails(strikes: np.ndarray, ivs: np.ndarray, S: float, T: float, r: float) -> tuple[np.ndarray, np.ndarray]`

Extends the strike/IV arrays into the tails using log-normal extrapolation.

- Left tail: extrapolate from the lowest 3 strikes using a linear fit in log-strike space; extend down to `S * 0.3` (70% below spot)
- Right tail: same logic; extend up to `S * 3.0` (200% above spot)
- Combine with the original data and return the extended arrays sorted by strike
- This is important: without tails, the RND will not integrate to 1 and Above bets near the current price will be mispriced

### `iv_smile_to_call_prices(strike_grid: np.ndarray, iv_spline: CubicSpline, S: float, T: float, r: float) -> np.ndarray`

Evaluates the spline on a dense grid and converts to call prices.

- Create a uniform strike grid from `min(strikes)` to `max(strikes)` with `STRIKE_GRID_POINTS` points
- Evaluate `iv_spline(strike_grid)` to get IVs on the grid
- Call `vectorized_call_price(S, strike_grid, T, r, iv_grid)` to convert
- Return `(strike_grid, call_prices)`

### `breeden_litzenberger(strike_grid: np.ndarray, call_prices: np.ndarray, r: float, T: float) -> RNDResult`

The main extraction function.

- Fit a second spline through the smoothed call prices (not the IVs — you already have smooth call prices from the previous step)
- Evaluate the **second derivative** of this spline at each grid point: `spline.derivative(2)(strike_grid)`
- Scale by `e^(rT)` to get the density: `rnd = np.exp(r * T) * second_deriv`
- Enforce non-negativity: `rnd = np.maximum(rnd, 0)` — negative densities are a numerical artefact
- Normalise: `rnd = rnd / np.trapz(rnd, strike_grid)` so it integrates to 1
- Compute diagnostics:
  - `integral`: should be ~1.0 before normalisation (if very far from 1, the tail extrapolation may be bad)
  - `peak_strike`: `strike_grid[np.argmax(rnd)]` — should be near the forward price `S * e^(rT)`
  - `left_mass`: mass below 50% of spot — sanity check for tail behaviour
- Return as `RNDResult`

### `extract_rnd(chain: OptionChain, r: float = RISK_FREE_RATE) -> RNDResult`

Top-level function that wraps the full pipeline from raw chain to RND.

1. Filter chain to calls only
2. Remove deep ITM and deep OTM options (keep strikes within `[0.5·S, 2.5·S]`) — these tend to have stale quotes
3. Call `fit_iv_smile` → `extrapolate_tails` → `iv_smile_to_call_prices` → `breeden_litzenberger`
4. Return `RNDResult`

**Test requirement:** Write tests in `tests/test_rnd.py` that verify:
- RND integrates to 1.0 (within 0.01 tolerance) using `np.trapz`
- RND is non-negative everywhere
- Peak of RND is within 10% of the forward price `S * e^(rT)`
- An "above ATM" probability is between 0.3 and 0.7 (sanity bound)

---

## 10. Module: `pricing/contracts.py`

Given a `RNDResult` and a `PolymarketContract`, compute the RND-implied fair value.

**Functions to implement:**

### `price_above(rnd: RNDResult, strike: float) -> float`

Probability that BTC ends above `strike` at expiry.

```python
mask = rnd.strikes >= strike
return np.trapz(rnd.density[mask], rnd.strikes[mask])
```

### `price_range(rnd: RNDResult, low: float, high: float) -> float`

Probability that BTC ends between `low` and `high`.

```python
mask = (rnd.strikes >= low) & (rnd.strikes <= high)
return np.trapz(rnd.density[mask], rnd.strikes[mask])
```

### `price_contract(contract: PolymarketContract, rnd: RNDResult) -> PricingResult`

Dispatcher that routes to the correct pricing function based on `contract.bet_type`.

- `"above"`: call `price_above(rnd, contract.strike_low)`
- `"range"`: call `price_range(rnd, contract.strike_low, contract.strike_high)`
- `"reach"` / `"dip"`: return a `PricingResult` with `rnd_fair_value = None` and a note that path-dependent pricing is not yet implemented — do not crash
- Compute `deviation = contract.polymarket_price - rnd_fair_value`
- Leave `band_lower`, `band_upper`, `violation` as `None` for now (filled in Stage 5)
- Return fully populated `PricingResult`

### `price_all_contracts(contracts: list[PolymarketContract], rnd: RNDResult) -> pd.DataFrame`

Maps `price_contract` over a list and returns results as a DataFrame. Sort by `abs(deviation)` descending.

---

## 11. Module: `analysis/band.py`

This is Stage 5 — implement it last, only after the pricing pipeline is validated end-to-end.

The no-arbitrage band accounts for the frictions that make exact replication impossible in practice.

**Functions to implement:**

### `call_spread_bounds(rnd: RNDResult, strike: float, deribit_strikes: np.ndarray) -> tuple[float, float]`

Approximates the digital option price using a call spread on the nearest traded strikes.

A digital call paying $1 if `S_T > K` can be approximated by: buy 1/(K₂-K₁) calls at K₁, sell 1/(K₂-K₁) calls at K₂, where K₁ < K < K₂ are the nearest Deribit strikes.

- `lower_bound`: use the tighter spread (K₁, K₂ closest to K)
- `upper_bound`: use the wider spread (next strikes out)
- Returns `(lower, upper)` in probability terms

### `transaction_cost_adjustment(num_options: int = 4) -> float`

Estimates the total transaction cost as a probability-equivalent deduction.

- `num_options`: approximate number of option legs in the replicating portfolio
- Each leg costs half the bid-ask spread. Use a conservative estimate of 0.5 vol points in IV terms, converting to price using vega
- Add `POLYMARKET_FEE` and `GAS_COST_USD / contract_notional` (assume $1000 notional)
- Returns a single float representing the total friction as a fraction of notional

### `compute_band(contract: PolymarketContract, rnd: RNDResult, deribit_strikes: np.ndarray) -> tuple[float, float]`

Combines call-spread bounds and transaction costs into the final no-arbitrage band.

```
lower = call_spread_lower - transaction_cost
upper = call_spread_upper + transaction_cost
```

Clip to `[0.0, 1.0]`.

### `check_violation(pricing_result: PricingResult) -> str | None`

- Returns `"above_band"` if `polymarket_price > band_upper`
- Returns `"below_band"` if `polymarket_price < band_lower`
- Returns `None` if inside the band

### `run_band_analysis(results: pd.DataFrame, rnd: RNDResult, deribit_strikes: np.ndarray) -> pd.DataFrame`

Iterates over a pricing results DataFrame, computes the band for each row, adds `band_lower`, `band_upper`, `violation` columns, and returns the augmented DataFrame.

---

## 12. Entry Point: `scripts/run_pipeline.py`

This script ties everything together into a single executable run. It must be runnable with `python scripts/run_pipeline.py` from the project root.

```python
"""
Full pipeline run:
  1. Fetch Deribit option chains for all active Polymarket contract expiries
  2. Extract RND for each expiry
  3. Fetch Polymarket BTC contracts
  4. Price each contract against the nearest RND
  5. Compute no-arb bands and flag violations
  6. Save results to data/snapshots/results/
  7. Print summary table to stdout
"""

import pandas as pd
import logging
from datetime import datetime

from config.settings import RISK_FREE_RATE
from data.fetchers.deribit import get_option_chain, find_nearest_expiry
from data.fetchers.polymarket import get_btc_contracts
from data.storage.local import save_option_chain, save_rnd
from pricing.rnd import extract_rnd
from pricing.contracts import price_all_contracts
from analysis.band import run_band_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def main():
    timestamp = int(datetime.utcnow().timestamp())
    
    log.info("Fetching Polymarket contracts...")
    contracts = get_btc_contracts()
    log.info(f"Found {len(contracts)} contracts")

    # Group contracts by expiry to avoid redundant Deribit fetches
    expiry_groups = {}
    for contract in contracts:
        nearest = find_nearest_expiry(contract.resolution_timestamp)
        expiry_groups.setdefault(nearest, []).append(contract)

    all_results = []

    for expiry_ts, group_contracts in expiry_groups.items():
        log.info(f"Processing expiry {expiry_ts} ({len(group_contracts)} contracts)")
        
        chain = get_option_chain(expiry_ts)
        if chain is None:
            log.warning(f"Could not fetch chain for expiry {expiry_ts}, skipping")
            continue
        
        save_option_chain(chain, label=f"_{timestamp}")
        
        rnd = extract_rnd(chain, r=RISK_FREE_RATE)
        save_rnd(rnd, label=f"_{timestamp}")
        
        log.info(f"RND diagnostics: {rnd.diagnostics}")
        
        results_df = price_all_contracts(group_contracts, rnd)
        
        deribit_strikes = chain.quotes["strike"].values
        results_df = run_band_analysis(results_df, rnd, deribit_strikes)
        
        all_results.append(results_df)

    if not all_results:
        log.error("No results produced. Check API connectivity.")
        return

    final_df = pd.concat(all_results).reset_index(drop=True)
    
    output_path = f"data/snapshots/results/pricing_{timestamp}.parquet"
    final_df.to_parquet(output_path, index=False)
    log.info(f"Results saved to {output_path}")
    
    # Summary to stdout
    print("\n=== PRICING SUMMARY ===\n")
    display_cols = ["contract_id", "bet_type", "polymarket_price", "rnd_fair_value",
                    "deviation", "band_lower", "band_upper", "violation"]
    print(final_df[display_cols].to_string(index=False))
    
    violations = final_df[final_df["violation"].notna()]
    print(f"\n{len(violations)} band violations detected out of {len(final_df)} contracts")

if __name__ == "__main__":
    main()
```

---

## 13. Notebooks

Three notebooks live in `notebooks/`. Their purpose is **validation and exploration**, not production logic. Never put business logic in notebooks — only calls to the modules above and visualisation code.

### `01_data_exploration.ipynb`

Purpose: understand the raw data before building on it.

What to do:
- Fetch one option chain manually and print the raw API response
- Plot the IV smile: x-axis = strike, y-axis = mid IV, separate series for calls and puts
- Plot bid-ask spread in IV terms vs strike — this shows where liquidity is thin
- Print a summary table: number of strikes, min/max strike, underlying price, time to expiry

### `02_rnd_validation.ipynb`

Purpose: verify that the RND extraction is working correctly.

What to do:
- Load a saved chain snapshot (not a live fetch)
- Run `extract_rnd` and plot the resulting density: x-axis = strike, y-axis = density
- Overlay a log-normal density with the same mean and variance for comparison
- Plot the cumulative distribution: `np.cumsum(rnd) * dK` — this should go from 0 to 1 smoothly
- Print all diagnostics
- Try deliberately breaking it: what happens if you use only 3 strikes? Only calls above ATM?

### `03_contract_pricing.ipynb`

Purpose: end-to-end pricing walkthrough for a single contract.

What to do:
- Pick one real Polymarket Above contract
- Load/fetch the matching RND
- Plot the RND with a vertical line at the contract's strike
- Shade the area under the density to the right of the strike — this is the RND fair value
- Print: `RND fair value = X, Polymarket price = Y, Deviation = Z`
- Run the band analysis and print whether it's a violation

---

## 14. Testing

Run the test suite with `pytest tests/` from the project root.

Priority order for writing tests:

1. `test_black_scholes.py` — write this first; the rest depends on BS being correct
2. `test_rnd.py` — use synthetic data (a log-normal distribution with known parameters) to test round-trip accuracy
3. `test_contracts.py` — test that `price_above(rnd, -inf) ≈ 1.0` and `price_above(rnd, +inf) ≈ 0.0`

For `test_rnd.py` the synthetic test works as follows: generate a log-normal distribution, compute call prices from it analytically, pass those prices through Breeden-Litzenberger, and verify the recovered density is close to the known input. This tests the numerical accuracy of the extraction independently of any API data.

---

## 15. Delivery Milestones

Complete these in order. Do not move to the next milestone until the current one passes its acceptance criteria.

| Milestone | Deliverable | Acceptance Criteria |
|---|---|---|
| **M1** | Data layer | `get_option_chain()` returns a clean DataFrame with ≥8 strikes for at least 2 expiries. Saved to parquet. |
| **M2** | Polymarket fetcher | `get_btc_contracts()` returns ≥3 parsed contracts with correct `bet_type` and numeric strikes |
| **M3** | Black-Scholes | All 3 BS tests pass. Round-trip IV test within 1e-6. |
| **M4** | RND extraction | RND integrates to 1±0.01. Peak within 10% of forward. Notebook 02 plot looks like a smooth bell. |
| **M5** | Contract pricing | `price_above` + `price_range` tested against known synthetic inputs. Notebook 03 shows end-to-end for one live contract. |
| **M6** | No-arb band | `run_pipeline.py` runs end-to-end without error and prints a results table with band columns populated. |

---

## 16. Common Failure Modes to Watch For

These are the most likely places things will go wrong:

**RND goes negative or spiky** — almost always caused by sparse strikes forcing the spline into oscillation. Fix: add more tail extrapolation points, or switch to a monotone spline (`scipy.interpolate.PchipInterpolator`) in the affected region.

**RND does not integrate to 1** — tail extrapolation is not wide enough. Extend the tail range further (try `S * 0.1` to `S * 5.0`) and check whether the pre-normalisation integral improves.

**Polymarket price parser returns None for most contracts** — the question text format has changed. Print 10 raw questions and update the regex.

**RND peak is far from forward price** — usually caused by including deep ITM calls with stale quotes. Tighten the strike filter in `extract_rnd`.

**Band lower > band upper** — your transaction cost estimate exceeds the call-spread width. This means the contract is too close to a traded strike for call-spread replication to work. Flag these as "unreplicable" rather than crashing.
