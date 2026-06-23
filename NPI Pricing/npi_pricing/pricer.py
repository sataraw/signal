"""Top-level pricing: turn NPI crossing probabilities into a fair-value interval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bernoulli import bernoulli_crossing_probability
from .first_passage import first_passage_probability


@dataclass(frozen=True)
class OneTouchQuote:
    """A fair-value bracket for a one-touch contract, from both NPI engines."""

    spot: float
    barrier: float
    horizon: int
    up: bool
    payout: float

    # probabilities
    bernoulli_lower: float
    bernoulli_upper: float
    lattice_lower: float
    lattice_upper: float

    # prices (probability * payout * discount)
    price_lower: float
    price_upper: float

    @property
    def price_mid(self) -> float:
        return 0.5 * (self.price_lower + self.price_upper)

    @property
    def spread(self) -> float:
        return self.price_upper - self.price_lower

    def decision(self, market_price: float) -> str:
        """A coherent imprecise-probability trade rule against a quoted price."""
        if market_price < self.price_lower:
            return "BUY (market below lower fair value)"
        if market_price > self.price_upper:
            return "SELL (market above upper fair value)"
        return "NO TRADE (market inside fair-value bracket)"

    def __str__(self) -> str:
        side = "up" if self.up else "down"
        return (
            f"One-touch {side}-barrier  spot={self.spot:g}  barrier={self.barrier:g}  "
            f"horizon={self.horizon}\n"
            f"  NPI-Bernoulli prob : [{self.bernoulli_lower:.3f}, {self.bernoulli_upper:.3f}]\n"
            f"  NPI-lattice   prob : [{self.lattice_lower:.3f}, {self.lattice_upper:.3f}]\n"
            f"  Fair value (x{self.payout:g}): [{self.price_lower:.4f}, {self.price_upper:.4f}]"
            f"  mid={self.price_mid:.4f}  spread={self.spread:.4f}"
        )


def price_one_touch(
    prices: np.ndarray,
    spot: float,
    barrier: float,
    horizon: int,
    *,
    payout: float = 1.0,
    up: bool | None = None,
    rate: float = 0.0,
    dt: float = 1.0,
    n_states: int = 121,
    bernoulli_stride: int = 1,
    combine: str = "lattice",
) -> OneTouchQuote:
    """Price a one-touch contract with both NPI engines.

    The reported price bracket comes from ``combine``:
      - ``"lattice"`` (default): use the imprecise-Markov-chain bracket — it
        conditions on distance-to-barrier and is the production engine.
      - ``"bernoulli"``: use the model-free count baseline.
      - ``"union"``: widest bracket (min lower, max upper) across both engines —
        most conservative.

    ``rate``/``dt`` apply a simple discount factor exp(-rate * horizon * dt);
    for most prediction markets ``rate=0`` is appropriate.
    """
    if up is None:
        up = barrier >= spot

    bern = bernoulli_crossing_probability(
        prices, spot, barrier, horizon, up=up, stride=bernoulli_stride
    )
    latt = first_passage_probability(
        prices, spot, barrier, horizon, up=up, n_states=n_states
    )

    if combine == "lattice":
        p_lo, p_hi = latt.lower, latt.upper
    elif combine == "bernoulli":
        p_lo, p_hi = bern.lower, bern.upper
    elif combine == "union":
        p_lo = min(bern.lower, latt.lower)
        p_hi = max(bern.upper, latt.upper)
    else:
        raise ValueError("combine must be 'lattice', 'bernoulli' or 'union'")

    discount = float(np.exp(-rate * horizon * dt))

    return OneTouchQuote(
        spot=spot,
        barrier=barrier,
        horizon=horizon,
        up=up,
        payout=payout,
        bernoulli_lower=bern.lower,
        bernoulli_upper=bern.upper,
        lattice_lower=latt.lower,
        lattice_upper=latt.upper,
        price_lower=payout * discount * p_lo,
        price_upper=payout * discount * p_hi,
    )
