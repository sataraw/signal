"""Implied-vs-empirical wedge tester on live Polymarket BTC contracts.

For each contract it computes:
  - our NPI physical-probability band [L, U],
  - the wedge lambda implied by (our band -> market price),
  - the empirically plausible wedge range (Yang 2026),
and flags contracts whose implied wedge falls OUTSIDE the plausible range —
i.e. the market price cannot be explained by a normal risk premium given our
physical estimate.

Run:  python examples/wedge_test.py   (from the repo root, shared .venv)
"""

from __future__ import annotations

from datetime import datetime, timezone

from npi_pricing.bounds import npi_probability
from src.market_data import binance_closes, binance_spot
from npi_pricing.polymarket import fetch_btc_contracts
from npi_pricing.wedge import EmpiricalWedge, cross_sectional_lambda, wedge_test

N_STATES = 161
# Empirical-wedge source. "constant" = the robust Polymarket MLE 0.176 (95% CI),
# shrunk for liquid/short crypto (default, most defensible). "xsection" = the
# headline wedge tilted by each contract's volume/duration/extremity/life
# (Yang 2026, Table 6), anchored so it stays unit-safe.
EMP_MODE = "constant"
CONST_SHRINK = 0.6        # pull 0.176 toward 0 for liquid short-dated crypto


def empirical_for(contract, duration_days, life_fraction):
    if EMP_MODE == "constant":
        return EmpiricalWedge(point=0.176, ci=(0.123, 0.230), shrink=CONST_SHRINK)
    return cross_sectional_lambda(
        contract.yes_price,
        baseline=0.176 * CONST_SHRINK,
        volume=max(contract.volume, 1.0),
        duration_days=max(duration_days, 1.0),
        life_fraction=life_fraction,
        uncertainty=0.08,
    )


def main() -> None:
    print("Fetching live Polymarket BTC contracts ...")
    contracts = fetch_btc_contracts(min_price=0.02, max_price=0.98)
    print(f"  {len(contracts)} live contracts")

    spot = binance_spot("BTCUSDT")
    prices = binance_closes("BTCUSDT", "1d", 1000)
    print(f"  spot=${spot:,.0f}  history={prices.size}d  empirical-wedge mode={EMP_MODE}\n")

    now = datetime.now(timezone.utc)
    header = (f"{'contract':34s} {'mkt':>6} {'our p*[L,U]':>14} "
              f"{'implied lam':>14} {'emp lam':>14} {'verdict':>12} {'edge':>6}")
    print(header)
    print("-" * len(header))

    flagged = []
    for c in contracts:
        days = c.horizon_days(now)
        horizon = max(1, int(round(days)))
        try:
            phys = npi_probability(prices, spot, c.barrier, horizon,
                                   up=c.up, kind=c.kind, n_states=N_STATES)
        except Exception as e:  # noqa: BLE001
            print(f"{c.question[:34]:34s}  skip ({e})")
            continue

        # elapsed fraction of contract life is unknown without start date; use a
        # neutral mid-life prior for the time-decay term.
        emp = empirical_for(c, duration_days=days, life_fraction=0.5)
        cmp = wedge_test(c.yes_price, phys.lower, phys.upper, emp)

        label = (c.question.replace("Will Bitcoin ", "")
                 .replace("Will the price of Bitcoin ", "").rstrip("?")[:34])
        print(f"{label:34s} {c.yes_price:6.3f} "
              f"[{phys.lower:4.2f},{phys.upper:4.2f}]  "
              f"[{cmp.implied.lo:+5.2f},{cmp.implied.hi:+5.2f}]  "
              f"[{cmp.emp_lo:+5.2f},{cmp.emp_hi:+5.2f}]  "
              f"{cmp.verdict:>12} {cmp.price_edge:6.3f}")
        if cmp.verdict != "consistent":
            flagged.append((c, cmp))

    print("\n" + "=" * 60)
    print("Implied wedge OUTSIDE the empirically plausible range "
          "(not explained by a normal risk premium):\n")
    if not flagged:
        print("  none — every market price is consistent with a plausible wedge.")
    for c, cmp in sorted(flagged, key=lambda t: -t[1].price_edge):
        print(f"  {cmp.verdict:11s} {c.question}")
        print(f"      market={cmp.market:.3f}  our p*=[{cmp.p_lower:.2f},{cmp.p_upper:.2f}]"
              f"  implied lam={cmp.implied.mid:+.2f}  plausible=[{cmp.emp_lo:+.2f},"
              f"{cmp.emp_hi:+.2f}]  price edge={cmp.price_edge:.3f}  liq=${c.liquidity:,.0f}")


if __name__ == "__main__":
    main()
