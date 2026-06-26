"""Polymarket fetching: Gamma API for discovery + question parsing, CLOB for live prices."""
from __future__ import annotations

from typing import Optional
import time
import requests
import json
from datetime import datetime
from dataclasses import asdict
import re
import json
import requests
from datetime import datetime
from typing import Optional, Tuple, List
from dataclasses import asdict

from schemas import PolymarketContract

def retrieve_btc_price_from_binance(start_time: datetime) -> Optional[float]:
    print(f"JUST FOR TEST USE: Fake BTC price from Binance retrieval for start time: {start_time.isoformat()}")
    return 59159.64  # Placeholder for actual implementation

def parse_market_to_polymarket_contract(market: dict) -> PolymarketContract:

    question = market.get("question", "")
    # Polymarket uses conditionId for on-chain resolution
    contract_id = market.get("conditionId", market.get("id", ""))
    start_time_str = datetime.fromisoformat(market.get("startDate").replace('Z', '+00:00'))
    end_dt = None
    
    # Determine the Bet Type
    q_lower = question.lower()
    if "between" in q_lower or "range" in q_lower:
        bet_type = "range"
    elif "dip" in q_lower or "fall" in q_lower:
        bet_type = "dip"
    elif "reach" in q_lower or "hit" in q_lower:
        bet_type = "reach"
    elif re.search(r"\bup\b", q_lower) and re.search(r"\bdown\b", q_lower):
        bet_type = "UpDown"
        start_time_str = datetime.fromisoformat(market.get("eventStartTime").replace('Z', '+00:00'))
    elif "less than" in q_lower or "below" in q_lower or "under" in q_lower:
        bet_type = "below"
    else:
        bet_type = "above"  # Catch-all for above and up or down TODO: Possible make this more fine grained
        
    # Parse the strike prices
    if bet_type == "UpDown":
        s1, s2 = parse_strike_from_id(question=question, bet_type=bet_type, start_time= start_time_str)
    else:
        s1, s2 = parse_strike_from_id(question=question, bet_type=bet_type)
    
    # Map strikes based on the bet type to match the dataclass logic
    strike_low, strike_high = None, None
    if bet_type == "range":
        strike_low, strike_high = s1, s2
    elif bet_type == "dip":
        strike_low = s1
    elif bet_type == "reach":
        strike_high = s1
    elif bet_type == "below":
        strike_high = s1
    else:
        strike_low = s1
        
    end_date_str = market.get("endDate")
    resolution_ts = 0
    if end_date_str:
        try:
            end_dt = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
            resolution_ts = int(end_dt.timestamp())
        except ValueError:
            pass
            
    # Parse the Polymarket Price (usually the first outcome corresponds to "Yes" or "Up")
    outcome_prices_str = market.get("outcomePrices", "[]")
    polymarket_price = None
    try:
        prices = json.loads(outcome_prices_str)
        if prices:
            polymarket_price = float(prices[0])
    except (json.JSONDecodeError, ValueError, IndexError):
        pass
        
    # Build contract
    contract = PolymarketContract(
        contract_id=contract_id,
        question=question,
        bet_type=bet_type,
        strike_low=strike_low,
        strike_high=strike_high,
        resolution_timestamp=resolution_ts,
        polymarket_price=polymarket_price,
        contract_lifetime=(start_time_str, end_dt)
    )

    return contract


def get_btc_contracts() -> list[PolymarketContract]:
    """Active BTC price-prediction markets, parsed into PolymarketContract."""

    gamma_url = "https://gamma-api.polymarket.com/markets"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://polymarket.com/"
    }

    params = {
        "tag_id": 21,
        "active": "true",
        "closed": "false",
        "limit": 100,
        "offset": 0,
        "order": "volume",
        "ascending": "false"
    }

    valid_contracts = []
    now_ts = int(time.time())
    page = 0

    print("Fetching liquid BTC markets (Sorted by Volume)...")

    while True:
        # Failsafe to prevent hitting the hard 2000 offset cap
        if params["offset"] >= 2000:
            break

        time.sleep(0.1)
        response = requests.get(gamma_url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 422:
            break # Reached server offset limit
            
        response.raise_for_status()
        markets = response.json()

        if not markets:
            break

        page += 1
        print(f"Scanning page {page}...")

        zero_volume_hit = False

        for m in markets:
            # If volume is 0, we have passed all active trading, no intrest in prematurely opened markets.
            vol = float(m.get("volume", 0))
            if vol == 0:
                zero_volume_hit = True
                continue 

            slug = m.get("slug", "").lower()
            question = m.get("question", "")
            
            if not m.get("active") or m.get("closed"):
                continue

            # Filter crypto for only BTC tickers
            if "btc" in question.lower() or "bitcoin" in question.lower() or "btc" in slug:
                
                # Check expiration, some contracts didnt get close properly
                end_date_str = m.get("endDate")
                if not end_date_str:
                    continue
                    
                try:
                    dt = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                    end_ts = int(dt.timestamp())
                except ValueError:
                    continue

                if now_ts >= end_ts:
                    continue

                # Only keep the currently active 5,15,1h,... BTC markets, the prematurely created ones are of no interest
                duration = 0
                if "-5m-" in slug:
                    duration = 300
                elif "-15m-" in slug:
                    duration = 900
                elif "-1h-" in slug or "-60m-" in slug:
                    duration = 3600

                if duration > 0:
                    start_ts = end_ts - duration
                    # ONLY append if the current time is exactly inside the active trading window!
                    if start_ts <= now_ts < end_ts:
                        valid_contracts.append(parse_market_to_polymarket_contract(m))
                else:
                    #TODO: Decide if we really wanna also keep all non-HFT (Macro) markets
                    valid_contracts.append(parse_market_to_polymarket_contract(m))

        # If we hit zero volume markets on this page, stop paginating completely.
        if zero_volume_hit:
            print("Hit zero-volume. Stopping query...")
            break

        params["offset"] += params["limit"]
    
    with open("valid_btc_markets.json", "w") as f:
        json.dump([asdict(contract) for contract in valid_contracts], f, indent=2, default=str)

    return valid_contracts


def parse_strike_from_id(question: str, bet_type: str, start_time: Optional[datetime] = None) -> tuple[Optional[float], Optional[float]]:
    """Extract strike(s) from contract ID. (strike, None) for above/reach/dip,
    (low, high) for range, (None, None) on failure."""
    """Extract strike(s) from the market question string using regex."""
    #TODO: Figure out how to handle non-price markets as well as the dynamically given strikes (5, 15, etc.. Up/down markets)
    # Strip commas that are between numbers to make float conversion easy
    clean_q = re.sub(r'(?<=\d),(?=\d)', '', question)
    
    def to_float(val_str: str, mult: str) -> Optional[float]:
        try:
            val = float(val_str)
            mult = mult.lower()
            if mult == 'k': val *= 1000
            elif mult == 'm': val *= 1000000
            elif mult == 'b': val *= 1000000000
            return val
        except ValueError:
            return None

    if bet_type == "range":
        # Matches: "between $72000 and $74000"
        match = re.search(r'between\s+\$?([\d\.]+)([kKmMbB]?)\s+and\s+\$?([\d\.]+)([kKmMbB]?)', clean_q, re.IGNORECASE)
        if match:
            return to_float(match.group(1), match.group(2)), to_float(match.group(3), match.group(4))
    elif bet_type == "UpDown":
        # Matches: "up" and "down" keywords
        if start_time is not None:
            return retrieve_btc_price_from_binance(start_time), None
        else:
            return None, None 
    else:
        # Matches: "above $68000", "reach $80000", "dip to 25000", "greater than $74000", "hit $150k"
        pat = r'(?:above|reach|dip to|hit|greater than|less than)\s+\$?([\d\.]+)([kKmMbB]?)'
        match = re.search(pat, clean_q, re.IGNORECASE)
        if match:
            return to_float(match.group(1), match.group(2)), None
            
        # Fallback: Just grab the first obvious dollar/money amount if keywords missed
        fallback = re.search(r'\$\s*([\d\.]+)([kKmMbB]?)', clean_q)
        if fallback:
            return to_float(fallback.group(1), fallback.group(2)), None

    return None, None
