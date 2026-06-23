"""End-to-end demo: price a one-touch contract from synthetic history.

Run:  python -m examples.demo      (from the project root)
"""

from __future__ import annotations

import numpy as np

from npi_pricing import price_one_touch
from npi_pricing.data import make_sample_prices


def main() -> None:
    prices = make_sample_prices(n_days=750, s0=100.0, sigma=0.20, seed=7)
    spot = float(prices[-1])

    print(f"Synthetic history: {prices.size} days, spot = {spot:.2f}\n")

    # A 20-day contract paying $1 if the price touches a barrier 4% above spot.
    horizon = 20
    barrier = spot * 1.04

    quote = price_one_touch(prices, spot, barrier, horizon, payout=1.0)
    print(quote)

    market = 0.18
    print(f"\nMarket quotes {market:.2f} -> {quote.decision(market)}")

    # Show how the bracket tightens as the barrier sits closer to spot.
    print("\nBarrier sensitivity (20-day, payout $1):")
    print(f"{'barrier':>9} {'rel move':>9} "
          f"{'lattice prob':>20} {'bernoulli prob':>20}")
    for mult in (1.01, 1.02, 1.04, 1.06, 1.10):
        b = spot * mult
        q = price_one_touch(prices, spot, b, horizon, payout=1.0)
        print(f"{b:9.2f} {mult - 1:9.1%} "
              f"   [{q.lattice_lower:5.3f}, {q.lattice_upper:5.3f}]   "
              f"   [{q.bernoulli_lower:5.3f}, {q.bernoulli_upper:5.3f}]")


if __name__ == "__main__":
    main()
