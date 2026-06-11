"""Single source of truth for all constants. Never hardcode numbers elsewhere."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API ---
DERIBIT_BASE_URL = os.getenv("DERIBIT_BASE_URL", "https://www.deribit.com/api/v2/public")
POLYMARKET_CLOB_URL = os.getenv("POLYMARKET_CLOB_URL", "https://clob.polymarket.com")
POLYMARKET_GAMMA_URL = os.getenv("POLYMARKET_GAMMA_URL", "https://gamma-api.polymarket.com")

# --- Market ---
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", 0.05))
CURRENCY = "BTC"

# --- RND extraction ---
MIN_STRIKES = 8           # Minimum strikes needed for a valid smile fit
TAIL_PERCENTILE = 0.01    # Log-normal tail extrapolation cutoff
STRIKE_GRID_POINTS = 2000  # Resolution of the interpolated strike grid

# --- Storage (relative to project root) ---
SNAPSHOT_DIR = "data/snapshots"

# --- No-arb band ---
POLYMARKET_FEE = 0.02     # 2% taker fee
GAS_COST_USD = 1.0        # Approx Polygon gas cost per transaction
