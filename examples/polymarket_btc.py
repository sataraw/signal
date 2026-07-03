"""Test the NPI pricer on live Polymarket Bitcoin one-touch contracts.

For each contract it computes an NPI confidence band (the arbitrage bound) from
BTC price history and flags any contract whose market price falls outside it.

Run:  python examples/polymarket_btc.py   (from the repo root, shared .venv)
"""

from __future__ import annotations

from datetime import datetime, timezone

from npi_pricing.bounds import arbitrage_band
from src.market_data import binance_closes, binance_spot
from npi_pricing.polymarket import fetch_btc_contracts

ALPHA = 0.05          # -> 90% confidence band
N_BOOT = 150
N_STATES = 161
FEE = 0.0             # Polymarket taker fee (currently ~0)


def main() -> None:
    print("Fetching live Polymarket BTC contracts ...")
    contracts = fetch_btc_contracts(min_price=0.02, max_price=0.98)
    print(f"  {len(contracts)} live one-touch contracts\n")

    print("Fetching BTC spot + daily history (Binance) ...")
    spot = binance_spot("BTCUSDT")
    prices = binance_closes("BTCUSDT", "1d", 1000)
    print(f"  spot = ${spot:,.0f}   history = {prices.size} daily closes\n")

    now = datetime.now(timezone.utc)
    header = (f"{'contract':36s} {'kind':>8} {'days':>4} {'mkt':>6} "
              f"{'NPI[L,U]':>15} {'arb band':>15} {'signal':>9} {'edge':>6}")
    print(header)
    print("-" * len(header))

    flagged = []
    for c in contracts:
        days = c.horizon_days(now)
        horizon = max(1, int(round(days)))
        try:
            band = arbitrage_band(
                prices, spot, c.barrier, horizon,
                up=c.up, kind=c.kind, n_states=N_STATES, alpha=ALPHA,
                n_boot=N_BOOT, friction=c.half_spread + FEE,
                continuous_monitoring=True, vol_window=30,
            )
        except Exception as e:  # noqa: BLE001
            print(f"{c.question[:36]:36s}  skip ({e})")
            continue

        signal, edge = band.classify(c.yes_price)
        kind_lbl = f"{c.kind[:4]}/{'up' if c.up else 'dn'}"
        label = (c.question.replace("Will Bitcoin ", "")
                 .replace("Will the price of Bitcoin ", "").rstrip("?")[:36])
        print(f"{label:36s} {kind_lbl:>8} {days:4.1f} "
              f"{c.yes_price:6.3f} "
              f"[{band.npi_lower:4.2f},{band.npi_upper:4.2f}]   "
              f"[{band.arb_lower:4.2f},{band.arb_upper:4.2f}]   "
              f"{signal:>9} {edge:6.3f}")
        if signal != "no-trade":
            flagged.append((c, band, signal, edge))

    print("\n" + "=" * 60)
    if flagged:
        print(f"{len(flagged)} contract(s) outside the {int((1-2*ALPHA)*100)}% band "
              f"(potential edge, net of frictions):\n")
        for c, band, signal, edge in sorted(flagged, key=lambda t: -t[3]):
            print(f"  {signal:8s} {c.question}")
            print(f"           market={c.yes_price:.3f}  band=[{band.arb_lower:.3f},"
                  f"{band.arb_upper:.3f}]  edge={edge:.3f}  "
                  f"liq=${c.liquidity:,.0f}")
    else:
        print(f"No contracts outside the {int((1-2*ALPHA)*100)}% band — "
              f"market is consistent with NPI given the data.")


if __name__ == "__main__":
    main()
