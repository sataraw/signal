# NPI Pricing — barrier / one-touch prediction-market contracts

Price prediction-market contracts that **"pay $1 if a boundary is crossed"** using
**Nonparametric Predictive Inference (NPI)** — Coolen's imprecise-probability
framework built on Hill's A(n) assumption — and compare the result to live
Polymarket prices. This single document covers the idea, the method, the
architecture, the design decisions, where the work sits in the literature, and
the assumptions (including the hard-coded constants) you must understand before
trusting any signal it emits.

---

## 1. The problem it solves

The goal is to price **contracts of the form "pays \$1 if a boundary is crossed."**
In derivatives terms that is a **one-touch / American digital (barrier) option**,
and its value is one number: the **first-passage probability** — P(the underlying
touches the barrier before expiry).

The twist is the *method*. Instead of assuming a model (Black–Scholes, a fixed
volatility, a parametric distribution), we use NPI, which estimates probabilities
directly from historical data and returns a **lower/upper interval `[L, U]`**, not
a point. That interval is the natural raw material for a **bid/ask bracket** and an
**arbitrage bound**.

For prediction markets this fits well: you want the *real-world* (physical)
probability, not a risk-neutral one, and there is no replication — so the fair
value is essentially the physical probability of crossing, which is exactly what
NPI estimates. We price the *underlying's* history (BTC), **not** the historical
prices of the contracts themselves — see [§9](#9-where-this-sits-in-the-literature)
for why that is the correct primitive.

---

## 2. The core method (two engines)

| Engine | Idea | Output |
|---|---|---|
| `bernoulli` | Slide windows of length = horizon over history, count how many crossed the (relative) barrier. NPI-Bernoulli: `[s/(n+1), (s+1)/(n+1)]`. | Model-free baseline |
| `first_passage` | Imprecise Markov chain on a log-price lattice; transition probabilities from NPI on historical increments; barrier = absorbing set; backward-recurse lower/upper hitting probability. | Production engine — conditions on distance-to-barrier and time left |

### Engine 1 — NPI-Bernoulli (`bernoulli.py`), the blunt baseline

Slide a window of length = the contract horizon across history; count how many
windows crossed the barrier. If `s` of `n` crossed, NPI gives `[s/(n+1), (s+1)/(n+1)]`.
Fully model-free, no path model. Weakness: it ignores where the price sits now, and
overlapping windows inflate `n` (making the interval look artificially tight — use
`bernoulli_stride=horizon` for honest, independent windows).

### Engine 2 — Imprecise Markov chain on a lattice (`first_passage.py`)

This is the "NPI tree" and the production engine:

1. Discretize log-price onto a grid.
2. Take historical one-step log-returns. NPI's A(n) puts probability mass
   `1/(n+1)` on each of the `n+1` gaps between the ordered returns.
3. Within a gap the next price can land anywhere, so the **lower expectation**
   gives that gap's mass to the *worst* reachable next state (`min`), and the
   **upper expectation** to the *best* (`max`).
4. The barrier is an **absorbing set**; a backward recursion of these lower/upper
   expectations over the horizon yields the lower/upper one-touch probability.

The **lower** probability is the value an adversary drives *away* from the barrier;
the **upper** is driven *toward* it — a tight, coherent no-arbitrage `[bid, ask]`
bracket. The same recursion *without* absorption gives the **terminal**
(European-digital) probability, used for "above \$X on date" contracts.

---

## 3. Quickstart

All commands run from the **repo root** in the shared environment
(`python -m venv .venv && .venv/bin/pip install -r requirements.txt -e .`):

```bash
.venv/bin/python -m pytest tests/                  # whole suite (NPI + pipeline)
.venv/bin/python examples/demo.py                  # synthetic
.venv/bin/python examples/polymarket_btc.py        # live arbitrage band
.venv/bin/python examples/wedge_test.py            # live implied-vs-empirical λ
```

```python
from npi_pricing import price_one_touch
from npi_pricing.data import make_sample_prices

prices = make_sample_prices()          # or your own 1-D price history
spot   = float(prices[-1])

quote = price_one_touch(prices, spot, barrier=spot * 1.04, horizon=20, payout=1.0)
print(quote)                           # probability + fair-value brackets
print(quote.decision(market_price=0.18))   # BUY / SELL / NO TRADE
```

### How to read the output

- `price_lower / price_upper` — fair-value bracket (× payout, × discount).
- `price_mid` — quote midpoint; `spread` — width = your ambiguity-driven spread.
- `decision(market)` — coherent rule: buy below the lower bound, sell above the
  upper, otherwise stand aside.

---

## 4. Architecture

```
npi_pricing/
  bernoulli.py       NPI-Bernoulli baseline
  first_passage.py   ImpreciseLattice — touch + terminal pricing, grid sizing
  pricer.py          price_one_touch -> OneTouchQuote (synthetic-data convenience)
  bounds.py          arbitrage_band -> ConfidenceBand (bootstrap + BGK + vol scale)
  wedge.py           Wang transform: implied λ vs empirical λ (wedge_test)
  polymarket.py      fetch/parse live BTC contracts (touch vs terminal)
  data.py            synthetic sample prices
src/market_data.py   BTC spot + history (Binance) — shared pipeline data layer
examples/
  demo.py            synthetic end-to-end demo
  polymarket_btc.py  live Polymarket BTC test (arbitrage band)
  wedge_test.py      live implied-vs-empirical wedge test
tests/test_npi.py    11 tests (incl. a brute-force check of the lattice operator)
tests/test_ground_truth.py  GBM/MC ground-truth, convergence + coverage tests
```

Pure Python + NumPy, runs in a local `.venv`.

### Conceptual map (one picture)

```
historical BTC returns
        |
        v
  regime adjustment  (skew-aware up/down vol, zero drift)   [bounds._regime_adjust]
        |
        v
   NPI imprecise Markov chain on a log-price lattice         [first_passage]
   (barrier absorbing = one-touch ; no absorption = terminal)
        |
        v
  physical probability band  [L, U]                          [bounds.npi_probability]
        |
        +--> + bootstrap + BGK monitoring + frictions  =  arbitrage band   [bounds.arbitrage_band]
        |
        +--> implied wedge lambda = Phi^{-1}(market) - Phi^{-1}(p*)
                 compared to empirical lambda (Yang 2026)    [wedge.wedge_test]
                 -> consistent / too rich / too cheap
```

---

## 5. The arbitrage / deviation bound (`bounds.py`)

The raw NPI `[L, U]` only captures **model ambiguity**. To call a price *mispriced
enough to trade*, `arbitrage_band` widens it for three more real sources of
deviation:

1. **Estimation noise** — `[L, U]` comes from one finite history. A *moving-block
   bootstrap* (preserves volatility clustering, our main model-risk caveat) gives
   the sampling spread. The band uses **outer quantiles**: across `n_boot=200`
   resamples, each producing a `(L, U)` pair,

   ```
   arb_lower = percentile( L over bootstraps, 100*alpha )      - friction
   arb_upper = percentile( U over bootstraps, 100*(1-alpha) )  + friction
   ```

   "Outer" is the operative word: it takes the **α-quantile of the lower bounds**
   and the **(1−α)-quantile of the upper bounds** — pushing each side *outward*.
   Result clipped to `[0, 1]`. Block length defaults to `√n`.
2. **Monitoring** — Polymarket resolves on the live (≈continuous) exchange price,
   but a daily-step lattice checks once a day, understating touches. The
   **Broadie–Glasserman–Kou** continuity correction shifts the barrier inward by
   `0.5826 * sigma * sqrt(dt)` to compensate (touch contracts only).
3. **Frictions** — the bid/ask half-spread plus any fee.

Decision rule: market YES price below `arb_lower` → **BUY YES**; above `arb_upper`
→ **SELL YES**; inside → no edge. The confidence level `1 − 2α` controls the
false-positive rate. `npi_probability` is the lightweight single-shot version (no
bootstrap) returning just the physical `[L, U]`.

> **Important — the arbitrage band is *not* risk-adjusted.** It is a **physical**
> probability band. The risk premium / wedge (§7) is **never folded into**
> `arbitrage_band`; it lives only in the separate `wedge.py` diagnostic. So a
> BUY/SELL flag here means the market deviates from the *physical* probability —
> which a normal risk premium may fully explain. For risk-adjusted signals, read
> the wedge tester's `price_band` / verdict, not the arb band.

---

## 6. The skew-aware vol fix (`bounds.py`, `_regime_adjust`)

A long lookback overstates move sizes in a calm market and bakes in a past bull
run's fat up-tail, so the engine regime-matches history to the recent window
(`vol_window=30`, Hull–White volatility-adjusted historical simulation):

- **Skew-aware rescaling** (`skew_aware=True`). "Skew" here is **not** a third-
  moment statistic. The residuals are split into up-moves and down-moves and each
  tail is rescaled by its *own* **root-mean-square semi-deviation** ratio —
  `s_up = sqrt(mean(e₊²))_recent / sqrt(mean(e₊²))_full` and likewise `s_dn` — so a
  stale fat up-tail shrinks independently of the down-tail. The empirical *shape*
  within each tail (the NPI gap structure) is preserved; only each tail's scale and
  the overall drift are retargeted.
- **Drift retargeting** via `drift`, default `"zero"` (driftless martingale) — the
  neutral choice, extrapolating neither the stale bull trend (`"full"`) nor a noisy
  30-day trend (`"recent"`).

This fixed the **downside** calibration (dip-to strikes near spot now read
*consistent* with the market). The **upside** gap remained — and the drift
comparison proved that is a genuine *directional / vol-level disagreement*, not a
vol-shape bug: only a bearish drift collapses upside to market levels, and that
overcorrects the downside. So we fixed the artifact and *report* the disagreement
rather than engineering it away.

---

## 7. The wedge layer (`wedge.py`)

From Yang (2026, the `paper.pdf` in this repo): in an incomplete market the market
price ≠ physical probability. They are linked by a **Wang transform**,

```
p_mkt = Phi( Phi^{-1}(p*) + lambda )
```

where λ (the "wedge") is a risk-premium / favorite-longshot term. Yang's
Proposition 1 also gives a *no-arbitrage interval* — the real analogue of our NPI
`[L, U]` — and the wedge is a *selection rule* picking a point *inside* that
interval.

So `wedge.py` does **not** hard-shift by λ. It runs the **implied-wedge test**:
given our band `[L, U]` and market price `m`, the wedge needed to reconcile them is

```
lambda_implied in [ Phi^{-1}(m) - Phi^{-1}(U) ,  Phi^{-1}(m) - Phi^{-1}(L) ]
```

Then it asks whether that range overlaps an **empirically plausible** λ:

- overlap → price explained by a normal risk premium (no edge),
- implied entirely **above** plausible → too rich (SELL),
- implied entirely **below** plausible → too cheap (BUY).

This is the imprecise-probability-consistent way to use the wedge: never commit to
one λ, ask whether *any* plausible λ explains the price. The tester also emits a
**wedge-adjusted fair market-price band** (`price_band = wang_transform(band, λ)`) —
this *is* the risk-adjusted analogue of the arbitrage band, and the place to look
if you want risk-adjusted signals. It ships its own `Phi` / `Phi^{-1}` (Acklam, no
SciPy) and reports saturated bands (`p* → 0/1`) as `n/a`.

**Empirical λ source** (`EMP_MODE`): `constant` = the robust Polymarket MLE
λ̂≈0.176 (95% CI [0.123, 0.230]), shrunk for liquid/short crypto; `xsection` = that
headline tilted by volume/duration/extremity/elapsed-life (Yang Table 6),
**anchored** so it stays unit-safe — the paper's raw `ln(Volume)` term is unit-
sensitive and, applied literally, drives λ to nonsense, so we use the slopes only
as relative tilts around the baseline.

---

## 8. The live Polymarket test (`examples/polymarket_btc.py`)

Pulls current BTC contracts from Polymarket's Gamma API + BTC spot/history from
Binance, prices each, and flags any outside the band. `polymarket.py` distinguishes
**two contract types**, because they price differently:

- "reach / dip to \$X" → **one-touch** barrier (path-dependent),
- "above / below \$X on \<date\>" → **terminal** European digital (endpoint only;
  pricing these as one-touch is wrong — spot already past the level gives a trivial
  1.0).

**Validation:** terminal digitals near spot reproduce the market within the band
(e.g. "above \$62k on June 19": market 0.72 vs NPI [0.71, 0.71]) — independent
confirmation that the engine is sane.

**What the flags mean.** The remaining flags are mostly **interpretable model/market
effects**, not clean arbitrage:

- The persistent "SELL" on deep downside "dip to" strikes is the model missing the
  market's downside skew / crash premium.
- The persistent upside "reach" gap is the directional/vol-level disagreement of §6.
- Tiny intraday flags come from **rounding sub-day horizons up to one step** — e.g.
  a 0.4-day "dip to" contract priced as a full daily step can throw a large false
  BUY. Treat these as artifacts, not edges.

Treat the bound as a **disagreement detector**: a breach flags either an edge **or**
a model assumption that doesn't hold for that contract — and tells you which.

---

## 9. Where this sits in the literature

This project is an **original synthesis of well-established techniques layered on
one specific recent paper** — *not* a source of new empirical findings. Almost every
building block is already published; the empirical conclusions largely reproduce
Yang (2026). Honest prior-art accounting:

**Already established (not new here):**

- **NPI applied to option pricing** is an existing sub-literature (Coolen's group,
  Durham): the idea that NPI on the underlying yields an *interval* option price
  whose bounds are the maximum buying / minimum selling price is published for
  [European/American binomial-tree options](https://www.tandfonline.com/doi/full/10.1080/03610926.2020.1764040)
  and for [Asian options](https://www.sciencedirect.com/science/article/abs/pii/S0927538X23000070)
  ([arXiv 2008.13082](https://arxiv.org/pdf/2008.13082);
  [NPI thesis index](https://npi-statistics.com/phdtheses.html)).
- **The first-passage engine is a textbook imprecise-Markov-chain computation.**
  Lower/upper hitting probabilities of an absorbing set — exactly what
  `first_passage.py` backward-recurses — are a named, published object:
  [Krak et al. 2019](http://proceedings.mlr.press/v103/krak19a/krak19a.pdf),
  De Bock/De Cooman, and as recently as
  [arXiv 2512.16696 (Dec 2025)](https://arxiv.org/pdf/2512.16696)
  ([intro](https://link.springer.com/chapter/10.1007/978-3-030-60166-9_5)).
- **The wedge half is the bundled Yang (2026) paper** ("Pricing Prediction Markets:
  Incomplete Markets, Selection Rules, and Risk Premia", `paper.pdf`), plus decades
  of **favorite-longshot** work (Wang transform: Wang 2000/2002; bias:
  [NBER w15923](https://www.nber.org/system/files/working_papers/w15923/w15923.pdf),
  Snowberg & Wolfers, Manski). The empirical "findings" this repo reports — positive
  wedge on real-money venues, favorite-longshot, downside skew, a single constant λ
  not fitting — are **downstream of / replicate Yang**, not independent discoveries.
- **Corrections are standard:** Broadie–Glasserman–Kou continuity correction (1997),
  Hull–White vol-adjusted historical simulation (1998).
- **Concurrent parallel effort:** treating prediction-market contracts as options is
  in the air — see [arXiv 2510.15205, "Toward Black–Scholes for Prediction Markets"
  (Oct 2025)](https://arxiv.org/pdf/2510.15205) (different machinery, same ambition).

**What appears genuinely unpublished (the synthesis):**

1. NPI imprecise pricing applied specifically to **barrier / one-touch / first-
   passage** contracts (existing NPI-option work is European/American/Asian).
2. Applying it to **crypto prediction-market** contracts (touch vs. terminal split).
3. Using the NPI `[L, U]` band as the **no-arbitrage selection interval** of Yang's
   framework and running an *implied-λ-vs-empirical-λ overlap* test rather than
   committing to a single λ.

So: new as an **engineering combination**, publishable at most as an applied note,
and only with full citation of the prior art above. Yang's paper (April 2026)
postdates this author's reference knowledge, so for the very latest follow-on work,
treat the citation list as a starting point rather than exhaustive.

---

## 10. Hard-coded values & assumptions (read before trusting any signal)

The **probability *engine* is genuinely data-driven** — NPI learns the whole return
distribution from data, with no parametric family — and it validates well near spot.
But the **trade *signals* depend materially on hard-coded choices and structural
assumptions.** Know these before acting on a flag:

**Structural assumptions (not just a knob):**

- **iid increments / random walk.** The lattice applies NPI to one-step returns
  independently. Real returns have **volatility clustering / autocorrelation**; this
  is *mitigated* (regime matching, block bootstrap) but not modeled away. Deepest
  limitation.
- **Daily monitoring + sub-day rounding.** Horizons round **up to one full step**, so
  sub-day contracts (e.g. 0.4 days) are systematically over-priced and throw false
  near-term BUYs. Mitigated for full-day barriers by the BGK correction.
- **Grid discretization.** Off-grid moves clamp to the boundary; the barrier is
  monitored per step (discrete), with the upper bound capturing intra-step touches.

**Hard-coded constants (defaults that change the output):**

| Constant | Default | Where | Note |
|---|---|---|---|
| `drift` | `"zero"` | `bounds._regime_adjust` | **The big one** — the whole upside disagreement flips under `"full"`/bearish drift. |
| `vol_window` | `30` | `bounds.*` | Recent-regime window for vol/skew matching. |
| BGK β | `0.5826` | `bounds._BGK_BETA` | Continuity-correction constant; correction always on for touch. |
| `n_states` | `121` | lattice | Grid resolution (accuracy ↔ speed). |
| `pad_sigmas` | `4.0` | `grid_bounds` | Grid roaming pad. |
| `alpha` | `0.05` | `arbitrage_band` | Band tail probability (⇒ 90% band). |
| `n_boot` | `200` | `arbitrage_band` | Bootstrap resamples; `block=√n`, `seed=12345`. |
| Empirical λ | `0.176`, CI `[0.123, 0.230]` | `wedge.EmpiricalWedge` | Yang headline MLE; hand-shrunk for crypto. |
| X-section slopes | `b_vol=-0.057, b_dur=0.109, b_ext=-0.290, g1=-0.156, g2=0.074` | `wedge._XS` | Yang Table 6; **re-anchored** because the literal coefficients are unit-sensitive and "drive λ to nonsense". |
| `vol_ref / dur_ref / _EXT_REF / uncertainty` | `100k / 30 / 0.25 / 0.08` | `wedge` | Reference points for the tilts. |
| `rate` | `0` | pricer | No discounting (fine for most prediction markets). |

**Bottom line.** The probability estimate is about as assumption-light as you can
get and produces sane, validated numbers near spot. The probability-to-signal
*pipeline* is not: it leans on the zero-drift choice, the sub-day rounding, and a
hand-anchored empirical wedge. Read this as a **disagreement detector** (its stated
purpose) and it is honest and useful — it tells you *where* model and market diverge
and usually *why*. Read it as a turnkey arbitrage signal and several live flags are
artifacts (sub-day rounding) or unadjusted risk premium (physical vs. risk-adjusted,
§5), not tradeable edges.

---

## 11. Other modelling caveats

- **Overlapping windows inflate `n`.** With `stride=1` the Bernoulli interval
  `1/(n+1)` looks dishonestly tight; use `bernoulli_stride=horizon` for honest,
  independent windows.
- **The empirical λ is noisy and estimand-mismatched** — used as a loose
  plausibility envelope, not a precise shift.
- **Price-threshold markets only.** This approach needs a tradeable underlying with a
  price history (it prices BTC's returns). It cannot price election/event markets,
  which have no such underlying.

---

## 12. What we learned

- The engine is **validated** where its assumptions hold (terminal digitals near
  spot match the market).
- A **single constant wedge λ does not fit** BTC contracts — the implied wedge swings
  and flips sign with direction.
- The remaining flags are mostly **interpretable model/market effects** (favorite-
  longshot, drift disagreement, sub-day rounding), not clean arbitrage. The whole
  stack is best read as a **disagreement detector**: a breach means either a real
  edge *or* an assumption that fails for that contract — and it tells you which.
</content>
</invoke>
