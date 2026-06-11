"""Full pipeline entry point (wired up at M6).

  1. Fetch Polymarket BTC contracts
  2. Group by nearest Deribit expiry; fetch + snapshot each option chain
  3. Extract RND per expiry
  4. Price each contract; compute no-arb bands; flag violations
  5. Save results and print summary
"""
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    raise NotImplementedError("Wired up at M6")


if __name__ == "__main__":
    main()
