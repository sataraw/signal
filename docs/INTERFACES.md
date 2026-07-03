# Interfaces: contract in, price history in, quote interval out

Phase 1.3 deliverable (PLAN.md). Defines the data contracts between the team
pipeline (`src/`), the NPI pricers (`npi_pricing/`), and teammates' components.
This doc doubles as the schema contribution to the scope document.

## 1. Contract spec (in)

Canonical dataclass: **`schemas.PolymarketContract`** (repo root — alignment
target with Awar's matching work on `polymarket_data_pull`):

| field | type | meaning |
|---|---|---|
| `contract_id` | str | Polymarket market id / slug |
| `question` | str | raw question text (source of truth for parsing) |
| `bet_type` | str | `"above"` \| `"range"` \| `"reach"` \| `"dip"` |
| `strike_low` | float? | strike (Above/Reach/Dip) or lower bound (Range) |
| `strike_high` | float? | upper bound (Range only) |
| `resolution_timestamp` | int | s epoch of resolution |
| `polymarket_price` | float? | YES mid price ∈ [0, 1] |

The NPI fetcher (`npi_pricing.polymarket.Contract`) uses a different but
losslessly mappable vocabulary — **path-dependence** is the primary split,
matching the lattice's two recursions:

| `bet_type` (pipeline) | NPI `kind` | NPI `up` | recursion |
|---|---|---|---|
| `reach` | `touch` | True | absorbing barrier |
| `dip` | `touch` | False | absorbing barrier |
| `above` | `terminal` | True | no absorption (European digital) |
| `range` | `terminal` ×2 | — | P(low ≤ S_T ≤ high) = P(above low) − P(above high) |

**Open item (for Awar):** two Polymarket fetchers exist —
`src/polymarket.py` (CLOB, M2 stub, unimplemented) and
`npi_pricing/polymarket.py` (Gamma API, working, richer fields: bid/ask,
liquidity, volume, endDate). Proposal: implement M2 as a thin adapter that
returns `schemas.PolymarketContract` built on the Gamma fetcher, rather than a
parallel CLOB implementation. Decide together with the matching-schema freeze.

## 2. Price history (in)

Canonical shape: **1-D `np.ndarray` of chronological closes at a fixed
interval**, plus a spot float. Nothing downstream accepts anything else.

Current source: `src/market_data.py` (Binance klines) —
`binance_closes(symbol="BTCUSDT", interval="1d", limit=1000) -> np.ndarray`,
`binance_spot(symbol) -> float`.

Unit convention (important): the NPI `horizon` argument counts **steps of the
history interval**. Daily history → horizon in days; hourly history → horizon
in hours. Mixing intervals between history and horizon is the most likely
silent-error path — the backtest harness (Phase 5) should carry the interval
alongside the array.

**Blocked:** the team's GitHub data pipeline for minute-level BTC history
(URL/branch still TBD). When it lands, it must be wrapped in `src/market_data`
returning the same shape; consumers don't change.

## 3. Quote interval (out)

NPI engines return **intervals**, the RND pipeline returns **points**; the
Phase 4 comparison framework treats a point as a degenerate interval.

- `npi_pricing.first_passage.FirstPassageResult` — raw ambiguity bracket
  `[lower, upper]` (+ `midpoint`, `imprecision`). *Not a confidence interval* —
  see `docs/npi/ASSUMPTIONS.md` #6.
- `npi_pricing.bounds.ConfidenceBand` — the actionable object:
  `npi_lower/npi_upper` (full-sample bracket), `arb_lower/arb_upper`
  (bootstrap + friction band, ~90% coverage at alpha=0.05),
  `classify(market_price) -> (signal, edge)`.
- `npi_pricing.pricer.OneTouchQuote` — fair-value bracket × payout, with
  `decision(market_price)`.
- `schemas.PricingResult` — the RND pipeline's per-contract row
  (`rnd_fair_value`, `deviation`, `band_lower/upper`, `violation`); the
  backtest results table (Phase 5) uses this shape for every pricer.

Phase 4 pricer interface (target): `price(contract, market_state) -> point or
interval`, each implementation carrying an `assumptions` metadata dict
(`measure`, `tail_model`, `jump_model`, `settlement`) per PLAN.md Phase 4.1.
