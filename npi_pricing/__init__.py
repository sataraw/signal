"""NPI Pricing — Nonparametric Predictive Inference for barrier / one-touch contracts.

Two engines for pricing a contract that "pays out if a boundary is crossed at
any point over the life of the contract":

1. ``bernoulli`` — NPI for Bernoulli data applied to historical crossing events.
   The blunt, fully model-free baseline. Returns a [lower, upper] probability.

2. ``first_passage`` — an imprecise Markov chain on a log-price lattice whose
   transition probabilities come from NPI (Hill's A(n)) applied to historical
   increments. Computes the lower/upper *first-passage* (one-touch) probability,
   conditioning on the current distance to the barrier and time remaining.

``pricer.price_one_touch`` ties both together into a fair-value interval.
"""

from .pricer import OneTouchQuote, price_one_touch
from .bernoulli import npi_bernoulli, bernoulli_crossing_probability
from .first_passage import first_passage_probability, ImpreciseLattice
from .bounds import ConfidenceBand, arbitrage_band, npi_probability
from .wedge import (
    EmpiricalWedge,
    ImpliedWedge,
    WedgeComparison,
    cross_sectional_lambda,
    implied_wedge,
    wang_transform,
    wedge_test,
)

__all__ = [
    "OneTouchQuote",
    "price_one_touch",
    "npi_bernoulli",
    "bernoulli_crossing_probability",
    "first_passage_probability",
    "ImpreciseLattice",
    "ConfidenceBand",
    "arbitrage_band",
    "npi_probability",
    "EmpiricalWedge",
    "ImpliedWedge",
    "WedgeComparison",
    "cross_sectional_lambda",
    "implied_wedge",
    "wang_transform",
    "wedge_test",
]
