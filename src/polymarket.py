"""Polymarket fetching: Gamma API for discovery + question parsing, CLOB for live prices."""
from __future__ import annotations

from typing import Optional

import re
import json
import requests
from datetime import datetime
from typing import Optional, Tuple, List

from schemas import PolymarketContract


def parse_market_to_polymaqket_contract(market: dict) -> PolymarketContract:

    question = market.get("question", "")
    # Polymarket uses conditionId for on-chain resolution
    contract_id = market.get("conditionId", market.get("id", ""))
    
    # Determine the Bet Type
    q_lower = question.lower()
    if "between" in q_lower or "range" in q_lower:
        bet_type = "range"
    elif "dip" in q_lower or "fall" in q_lower:
        bet_type = "dip"
    elif "reach" in q_lower or "hit" in q_lower:
        bet_type = "reach"
    else:
        bet_type = "above"  # Catch-all for above and up or down TODO: Possible make this more fine grained
        
    # Parse the strike prices
    s1, s2 = parse_strike_from_question(question)
    
    # Map strikes based on the bet type to match the dataclass logic
    strike_low, strike_high = None, None
    if bet_type == "range":
        strike_low, strike_high = s1, s2
    elif bet_type == "dip":
        strike_low = s1
    elif bet_type == "reach":
        strike_high = s1
    else:
        strike_low = s1
        
    end_date_str = market.get("endDate")
    resolution_ts = 0
    if end_date_str:
        try:
            dt = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
            resolution_ts = int(dt.timestamp())
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
        polymarket_price=polymarket_price
    )

    return contract


def get_btc_contracts() -> list[PolymarketContract]:
    """Active BTC price-prediction markets, parsed into PolymarketContract."""

    gamma_url = "https://gamma-api.polymarket.com/markets"  #Used for market exploration
    clob_url = "https://clob.polymarket.com/midpoint"   #Used for direct single market price fetching

    offset = 0

    #Make sure we only check for curretnly active markets 
    # TODO: figure out how we want to promt this, there are a large number of btc markets that are open and active but from my understanding we mainly are interested in the 5min ones for now (this query tackles those)
    params = {
        "active" : "true",
        "closed" : "false",
        "limit" : 100,
        "order" : "createdAt",
        "offset" : offset,
        "ascending" : "false",
    }   

    valid_contracts = []

    while offset < 1500:  # Arbitrary max offset to prevent infinite loop and getting rate limited
        response = requests.get(gamma_url, params=params, timeout=10)
        response.raise_for_status()
        markets = response.json()

        for m in markets:
            question = m.get("question", "")

            if ("BTC" in question or "Bitcoin" in question):
                valid_contracts.append(parse_market_to_polymaqket_contract(m))
        
        offset += 100
        params["offset"] = offset

        # Was used to understand prompting of the api, use for test cases to avoid rate limits
        #json.dump(valid_contracts, open("valid_contracts4.json", "w"), indent=2)

    return valid_contracts


def parse_strike_from_question(question: str) -> tuple[Optional[float], Optional[float]]:
    """Extract strike(s) from question text. (strike, None) for above/reach/dip,
    (low, high) for range, (None, None) on failure."""
    raise NotImplementedError("M2")
    

