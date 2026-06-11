"""Local snapshot persistence (parquet + sidecar JSON). All paths relative to project root."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import SNAPSHOT_DIR
from schemas import OptionChain, RNDResult


def _dir(kind: str) -> Path:
    """data/snapshots/<kind>/, created if missing."""
    path = Path(SNAPSHOT_DIR) / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_option_chain(chain: OptionChain, label: str = "") -> str:
    """Save quotes as parquet + metadata sidecar JSON. Returns the parquet filepath."""
    expiry_date = datetime.fromtimestamp(chain.expiry_timestamp / 1000, timezone.utc).strftime("%Y%m%d")
    stem = f"btc_chain_{expiry_date}_{chain.fetch_time}{label}"
    parquet_path = _dir("chains") / f"{stem}.parquet"
    chain.quotes.to_parquet(parquet_path, index=False)

    meta = {
        "expiry_timestamp": chain.expiry_timestamp,
        "underlying_price": chain.underlying_price,
        "fetch_time": chain.fetch_time,
    }
    parquet_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return str(parquet_path)


def load_option_chain(filepath: str) -> OptionChain:
    """Load parquet + sidecar JSON into an OptionChain."""
    path = Path(filepath)
    meta = json.loads(path.with_suffix(".json").read_text())
    return OptionChain(
        expiry_timestamp=meta["expiry_timestamp"],
        underlying_price=meta["underlying_price"],
        fetch_time=meta["fetch_time"],
        quotes=pd.read_parquet(path),
    )


def save_rnd(rnd: RNDResult, label: str = "") -> str:
    raise NotImplementedError("M4")


def load_rnd(filepath: str) -> RNDResult:
    raise NotImplementedError("M4")


def list_snapshots(kind: str) -> list[str]:
    """kind is 'chains' or 'rnds'. Returns sorted parquet snapshot filepaths."""
    return sorted(str(p) for p in _dir(kind).glob("*.parquet"))
