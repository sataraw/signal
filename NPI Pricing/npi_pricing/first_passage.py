"""Imprecise Markov chain first-passage pricer — the NPI "tree".

We discretise log-price onto a grid and treat "touch the barrier before T" as
first passage to an absorbing set in a Markov chain whose one-step transition
probabilities are *imprecise* and come from NPI.

NPI on the historical one-step log-returns (Hill's A(n)):
    the next return falls in each of the n+1 gaps between the ordered historical
    returns with probability 1 / (n + 1).
Within a gap the return — hence the next state — can be anywhere in that gap, so:
    lower expectation: give each gap's mass to the *worst* reachable next state,
    upper expectation: give each gap's mass to the *best* reachable next state.

Backward recursion of these lower/upper expectations over the absorbing barrier
yields the lower/upper one-touch probability. The lower probability is the value
of the imprecise chain that an adversary drives away from the barrier; the upper
is the value driven toward it — i.e. a tight no-arbitrage [bid, ask] bracket.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FirstPassageResult:
    lower: float
    upper: float

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def imprecision(self) -> float:
        return self.upper - self.lower


def grid_bounds(
    increments: np.ndarray,
    spot_log: float,
    barrier_log: float,
    horizon: int,
    *,
    pad_sigmas: float = 4.0,
) -> tuple[float, float]:
    """Log-price grid bounds sized to contain spot, barrier and plausible roaming.

    Exposed so a bootstrap can fix the grid across resamples for comparability.
    """
    increments = np.asarray(increments, dtype=float)
    sigma = float(np.std(increments)) if increments.size else 0.0
    pad = max(pad_sigmas * sigma * np.sqrt(max(horizon, 1)),
              0.5 * abs(barrier_log - spot_log))
    lo = min(spot_log, barrier_log) - pad
    hi = max(spot_log, barrier_log) + pad
    span = hi - lo
    return lo - 0.05 * span, hi + 0.05 * span


class ImpreciseLattice:
    """A log-price grid plus NPI gap geometry, reusable across many contracts."""

    def __init__(
        self,
        increments: np.ndarray,
        log_lo: float,
        log_hi: float,
        n_states: int,
    ):
        if n_states < 3:
            raise ValueError("n_states must be >= 3")
        if log_hi <= log_lo:
            raise ValueError("require log_hi > log_lo")

        self.increments = np.sort(np.asarray(increments, dtype=float))
        self.n_states = int(n_states)
        self.grid = np.linspace(log_lo, log_hi, self.n_states)
        self.h = self.grid[1] - self.grid[0]

        # NPI gaps over the ordered increments: (-inf, r1), (r1, r2), ..., (rn, +inf).
        # Each carries probability mass 1 / (n + 1). Convert each gap's [a, b]
        # bounds into integer index offsets on the grid, with the infinite tails
        # clamped to span the whole grid.
        r = self.increments
        n = r.size
        big = self.n_states - 1
        weight = 1.0 / (n + 1)

        # Each gap maps to an integer index-offset range on the grid. Many gaps
        # (especially with fine, high-frequency data) collapse onto the same
        # offset range; since the lower/upper expectation is linear in the gap
        # masses, we merge equal ranges and sum their weights. This turns an
        # O(n) inner loop into O(distinct offsets) without changing the result.
        merged: dict[tuple[int, int], float] = {}
        for k in range(n + 1):
            a = -np.inf if k == 0 else r[k - 1]
            b = np.inf if k == n else r[k]
            o_lo = -big if np.isneginf(a) else int(np.round(a / self.h))
            o_hi = big if np.isposinf(b) else int(np.round(b / self.h))
            o_lo = max(o_lo, -big)
            o_hi = min(o_hi, big)
            o_hi = max(o_hi, o_lo)
            key = (o_lo, o_hi)
            merged[key] = merged.get(key, 0.0) + weight
        self._offsets = list(merged.keys())
        self._weights = np.array([merged[k] for k in self._offsets])

    @classmethod
    def from_history(
        cls,
        increments: np.ndarray,
        spot_log: float,
        barrier_log: float,
        horizon: int,
        *,
        n_states: int = 121,
        pad_sigmas: float = 4.0,
    ) -> "ImpreciseLattice":
        """Build a lattice sized to comfortably contain spot, barrier and roaming."""
        lo, hi = grid_bounds(np.asarray(increments, dtype=float),
                             spot_log, barrier_log, horizon, pad_sigmas=pad_sigmas)
        return cls(increments, lo, hi, n_states)

    def index_of(self, log_price: float) -> int:
        return int(np.argmin(np.abs(self.grid - log_price)))

    def _expectation(self, f: np.ndarray, *, lower: bool) -> np.ndarray:
        """One NPI lower/upper expectation step of value vector ``f`` over the grid."""
        s = self.n_states
        out = np.zeros(s)
        base = np.arange(s)
        for (o_lo, o_hi), w in zip(self._offsets, self._weights):
            width = o_hi - o_lo + 1
            cols = np.arange(width)
            idx = np.clip(base[:, None] + o_lo + cols[None, :], 0, s - 1)
            vals = f[idx]
            reduced = vals.min(axis=1) if lower else vals.max(axis=1)
            out += w * reduced
        return out

    def first_passage(
        self, spot_log: float, barrier_log: float, horizon: int, *, up: bool
    ) -> FirstPassageResult:
        """Lower/upper probability of touching the barrier within ``horizon`` steps."""
        absorbing = self.grid >= barrier_log if up else self.grid <= barrier_log
        i0 = self.index_of(spot_log)

        def run(lower: bool) -> float:
            f = absorbing.astype(float)  # g_0: 1 on the barrier set, else 0
            for _ in range(horizon):
                f = self._expectation(f, lower=lower)
                f[absorbing] = 1.0  # barrier is absorbing (one-touch)
            return float(f[i0])

        return FirstPassageResult(lower=run(lower=True), upper=run(lower=False))

    def terminal(
        self, spot_log: float, barrier_log: float, horizon: int, *, up: bool
    ) -> FirstPassageResult:
        """Lower/upper probability the price is past the level *at* ``horizon``.

        European-digital analogue of ``first_passage``: the same NPI backward
        recursion but with no absorption, so only the terminal state matters.
        """
        region = self.grid >= barrier_log if up else self.grid <= barrier_log
        i0 = self.index_of(spot_log)

        def run(lower: bool) -> float:
            f = region.astype(float)
            for _ in range(horizon):
                f = self._expectation(f, lower=lower)
            return float(f[i0])

        return FirstPassageResult(lower=run(lower=True), upper=run(lower=False))


def first_passage_probability(
    prices: np.ndarray,
    spot: float,
    barrier: float,
    horizon: int,
    *,
    up: bool | None = None,
    n_states: int = 121,
) -> FirstPassageResult:
    """Convenience wrapper: NPI imprecise-Markov-chain one-touch probability.

    ``prices`` are historical levels; one-step log-returns are extracted and used
    as the NPI sample of increments.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.size < 3:
        raise ValueError("need at least 3 prices to form increments")
    increments = np.diff(np.log(prices))
    if up is None:
        up = barrier >= spot

    lattice = ImpreciseLattice.from_history(
        increments,
        spot_log=np.log(spot),
        barrier_log=np.log(barrier),
        horizon=horizon,
        n_states=n_states,
    )
    return lattice.first_passage(
        np.log(spot), np.log(barrier), horizon, up=up
    )
