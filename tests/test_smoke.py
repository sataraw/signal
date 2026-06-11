"""M0 scaffold check: every module imports and the shared schemas construct.

Math/IO tests arrive with their milestones (M3 black-scholes, M4 rnd, M5 contracts).
"""
import numpy as np
import pandas as pd

import config
import schemas
from src import bands, blackscholes, contracts, deribit, polymarket, rnd, storage  # noqa: F401


def test_config_constants():
    assert config.MIN_STRIKES == 8
    assert 0 < config.RISK_FREE_RATE < 1


def test_schemas_construct():
    chain = schemas.OptionChain(
        expiry_timestamp=1_900_000_000_000,
        underlying_price=65000.0,
        fetch_time=1_700_000_000,
        quotes=pd.DataFrame({"strike": [60000, 70000]}),
    )
    assert chain.time_to_expiry > 0

    rnd_res = schemas.RNDResult(
        strikes=np.array([1.0, 2.0]),
        density=np.array([0.5, 0.5]),
        expiry_timestamp=chain.expiry_timestamp,
        underlying_price=chain.underlying_price,
    )
    assert rnd_res.diagnostics == {}
