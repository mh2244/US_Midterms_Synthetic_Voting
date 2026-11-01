#### Preamble ####
# Purpose: Downloads PredictIt prediction-market prices for every 2026 Senate race we forecast,
#          plus the market on which party controls the Senate: the list of markets and, for each,
#          the daily open, high, low and closing price of every contract over the last 7 days.
#          Used by scripts/16f-compare_markets.py to set our forecast beside the market's.
# Author: Kexin (Coco) Liu
# Date: 29 October 2026
# License: MIT
# Pre-requisites: Internet access. No key is needed; PredictIt's public API is free for
#   non-commercial use with attribution ("Data from PredictIt", https://www.predictit.org).
# Any other information needed? Run from the project root:  python scripts/02c-download_predictit.py
#   PredictIt keeps only the last 7 days of daily prices, so run this within a week of each news
#   day (27, 29 and 31 October); every download is kept in its own dated folder, so earlier days
#   are never lost. Requests are spaced 3 seconds apart, because PredictIt refuses faster ones
#   (HTTP 429).
# Outputs: data/01-raw_data/markets/predictit_<UTC timestamp>/
#   all_markets.json   every open PredictIt market at download time
#   chart_<id>.json    daily prices for one Senate market
#   daily_prices.csv   the chart files stacked: market, state, contract, date and prices

#### Workspace setup ####
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import config

BASE = "https://www.predictit.org/api"
OUT = Path("data/01-raw_data/markets")
HEADERS = {"User-Agent": "STA2453 US_Midterms_Synthetic_Voting (non-commercial research)"}
PAUSE = 3                                 # seconds between requests
RACE = re.compile(r"Which party will win the 2026 US Senate (?:special )?election in (.+)\?")
CONTROL = "Which party will control the Senate after the 2026 election?"


def get_json(url, tries=5):
    """GET a PredictIt endpoint, waiting longer after each refusal (HTTP 429)."""
    for attempt in range(tries):
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(PAUSE * (attempt + 2))
    raise RuntimeError(f"{url} failed after {tries} tries (last status {r.status_code})")


#### Which markets ####
nominees = pd.read_csv(config.NOMINEES_FILE)
states = set(nominees["state"])
markets = get_json(f"{BASE}/marketdata/all/")["markets"]
wanted = {}
for m in markets:
    g = RACE.match(m["name"])
    if g and config.STATE_ABBR.get(g.group(1)) in states:
        wanted[m["id"]] = config.STATE_ABBR[g.group(1)]
    elif m["name"] == CONTROL:
        wanted[m["id"]] = "control"
missing = sorted(states - set(wanted.values()))
print(f"{len(wanted)} markets: {sum(v != 'control' for v in wanted.values())} of {len(states)} races"
      + (f"; no market for {missing}" if missing else ""))

#### Download ####
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
folder = OUT / f"predictit_{stamp}"
folder.mkdir(parents=True, exist_ok=True)
(folder / "all_markets.json").write_text(json.dumps(markets, indent=1))
rows = []
for market_id, state in wanted.items():
    time.sleep(PAUSE)
    chart = get_json(f"{BASE}/Public/GetMarketChartData/{market_id}?timespan=7d&maxContracts=6"
                     "&isTimespanInHours=false")
    (folder / f"chart_{market_id}.json").write_text(json.dumps(chart, indent=1))
    for p in chart:
        rows.append({"market_id": market_id, "state": state, "contract_id": p["contractId"],
                     "contract_name": p["contractName"], "date": p["dateString"],
                     "open": p["openSharePrice"], "high": p["highSharePrice"],
                     "low": p["lowSharePrice"], "close": p["closeSharePrice"],
                     "volume": p["tradeVolume"]})
prices = pd.DataFrame(rows)
prices.to_csv(folder / "daily_prices.csv", index=False)

#### Checks ####
assert len(prices), "No prices downloaded"
assert prices["close"].between(0, 1).all(), "A price is outside 0 to 1"
assert prices.groupby("market_id")["date"].nunique().ge(1).all()
print(f"Saved {len(prices):,} daily prices from {prices['date'].min()} to {prices['date'].max()} in {folder}")
