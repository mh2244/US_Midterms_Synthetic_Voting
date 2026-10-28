#### Preamble ####
# Purpose: Settings shared by scripts 06-13, so a number or model name is changed in one place.
# Author: Maggie Huang
# Date: 22 October 2026
# License: MIT
# Pre-requisites: None. Other scripts do `import config`; run them from the project root,
#   for example: python scripts/07-build_personas.py
# Note: this file never contains the API key. The key is read from the environment
#   variable OPENROUTER_API_KEY, or from a file called .env in the project root
#   (a line such as OPENROUTER_API_KEY=sk-or-...). Put .env in .gitignore.

import os
import re
from datetime import date
from pathlib import Path

#### Settings you are likely to change ####
SEED = 2026               # one seed for every random step, so reruns match
N_PER_STATE = 100           # personas per state. Test run = 5; the real run was planned at 100
N_REPEATS = 3             # times each prompt is sent to each model (planned: 3)
N_DRAWS = 1000            # simulated outcomes per state and model (same as the simulated data)
# SD of an error shared by all states in a draw (two-party share). Polls miss in the same
# direction across states (Shirani-Mehr et al. 2018), and the synthetic voters can lean one way
# overall. 0.03 (3 points) is an ASSUMPTION; script 11 can be rerun with --national-sd 0 / 0.02 /
# 0.04 for the sensitivity table in the paper.
NATIONAL_SD = 0.03

# Short names are used in the data; the values are the OpenRouter model ids.
# CHECK the ids on https://openrouter.ai/models before running; models can be renamed
# or withdrawn without notice, which is why the id the API returns is saved with every answer.
# Full run (from 2026-10-27): four cheap paid models from four different companies.
# Prices per 1M tokens (input / output) on OpenRouter, checked 2026-10-25.
# The test run (2026-10-21) used poolside/laguna-s-2.1:free and nvidia/nemotron-3-ultra-550b-a55b:free.
MODELS = {
    "haiku": "anthropic/claude-haiku-5.5",        # Anthropic, $0.10 / $0.50
    "luna": "openai/gpt-6-luna",                  # OpenAI, $0.10 / $0.50
    "deepseek": "deepseek/deepseek-v4.1-flash",   # DeepSeek, $0.30 / $1.20 ($0.15 / $0.60 off-peak)
    "gemini": "google/gemini-3.5-flash-lite",     # Google, $0.30 / $2.50
}
# OpenRouter list prices, US dollars per million tokens (input, output), checked 2026-10-25;
# used by scripts/16d-costs.py
PRICES = {"haiku": (0.10, 0.50), "luna": (0.10, 0.50), "deepseek": (0.30, 1.20), "gemini": (0.30, 2.50)}

# Party colours used by every figure: blue for Democrats, red for Republicans, grey for independents
PARTY_COLOURS = {"Democratic": "#2166ac", "Republican": "#b2182b", "Independent": "#808080"}
# Names shown on the website (tables and figures), in the same order
MODEL_LABELS = {"haiku": "Claude Haiku 5.5", "luna": "GPT-6 Luna", "deepseek": "DeepSeek V4.1 Flash",
                "gemini": "Gemini 3.5 Flash-Lite"}

# Reasoning ("thinking") setting sent with every request (script 09). Hidden reasoning tokens are
# billed as output, so it is switched off: the answers only need two or three sentences.
# Keys are the short names in MODELS; "default" covers any model not listed. None = send nothing
# and use the model's own default (write that choice in the paper).
# Gemini 3.5 Flash Lite refuses requests with reasoning disabled (HTTP 400 "Reasoning is mandatory"),
# so it gets the smallest effort level instead.
REASONING = {
    "default": {"enabled": False},
    "gemini": {"effort": "minimal"},
}

# Embedding model for SSR (script 10). Chosen 2026-10-26 after a check on 200 real Florida answers:
# of 159 clear vote statements, the local MiniLM model put 12 on the wrong side of 50 per cent
# (answers that named the opponent were pulled toward that opponent's anchor); Gemini Embedding 2
# put 1. Every run must use the same embedder. "local" needs: pip install sentence-transformers
EMBEDDER = "openrouter"
LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENROUTER_EMBED_MODEL = "google/gemini-embedding-2"   # the model Heath & Alexander used; --embedder openrouter
                                                      # $0.20 per million tokens (":batch" variant $0.10), 8,192-token limit
EMBED_DIMENSIONS = None   # e.g. 768 to shrink the vectors (the model allows 128-3072). None = model default (3072)
# Asymmetric embedding (answers as "queries", anchors as "documents"), as in Heath & Alexander 2026.
# HOW to ask for it through OpenRouter is NOT confirmed: check the Gemini Embedding 2 documentation
# (task types or text prefixes) and then fill these in. Empty strings = no asymmetric mode.
EMBED_QUERY_PREFIX = ""
EMBED_DOC_PREFIX = ""
SSR_TEMPERATURE = 0.25    # Heath & Alexander Methods use 0.25 (their Results say 0.2)

# Which run the scripts read and write (scripts 08-13 and 15). Each run gets its own folder, so
# the three news days and the no-news baseline never overwrite or mix with each other.
# Planned runs: "2026-10-27", "2026-10-29", "2026-10-31" (with that day's news summary) and
# "nonews" (baseline, same personas and questions, no news). Change it here, or set it for one
# command only:   RUN=2026-10-29 python scripts/09-query_models.py
RUN = os.environ.get("RUN", "2026-10-27").strip()
NEWS_RUN = RUN != "nonews"          # every dated run includes that day's news; "nonews" does not
# Date the news and polls were collected (write it in the paper). For "nonews" it is the date of
# the first news run, which shares its personas and polls.
AS_OF = RUN if NEWS_RUN else "2026-10-27"

#### Paths (relative to the project root) ####
DATA = Path("data")
SHARED_ANALYSIS = DATA / "02-analysis_data"
if RUN != "nonews":
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", RUN):
        raise ValueError("RUN must be nonews or YYYY-MM-DD")
    date.fromisoformat(RUN)
# Base directories hold shared inputs; RUN directories hold experiment outputs.
ANALYSIS = SHARED_ANALYSIS
RESULTS = DATA / "03-results"
RUN_DIR = ANALYSIS / f"run_{RUN}"
RUN_RESULTS = RESULTS / f"run_{RUN}"
# Figures live outside docs/: docs/ is Quarto's output folder and is rebuilt on every render.
FIGURES = Path("figures") / f"run_{RUN}"
# The website shows the latest news run from these fixed paths (filled by script 17).
LATEST_RESULTS = RESULTS / "latest"
LATEST_FIGURES = Path("figures") / "latest"
CES_FILE = DATA / "ces2024_senate_demographics.csv"
CANDIDATES_RAW = DATA / "senate_candidates_2026.csv"
NEWS_FILE = DATA / "news" / "news_context.csv"
NEWS_SUMMARIES_FILE = DATA / "news" / "persona_news_summaries.csv"
POLLS_FILE = DATA / "polling_data" / "02-analysis_data" / "senate_polls_2026_general.csv"

NOMINEES_FILE = SHARED_ANALYSIS / "senate_nominees_2026.csv"
CANDIDATES_CLEAN = SHARED_ANALYSIS / "candidates_clean.csv"
PERSONAS_FILE = SHARED_ANALYSIS / "personas.parquet"
PROMPTS_FILE = RUN_DIR / "prompts.parquet"
RESPONSES_FILE = RUN_DIR / "responses.parquet"
SCORES_FILE = RUN_DIR / "scores.parquet"
FORECAST_FILE = RUN_RESULTS / "forecast.parquet"

STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI",
    "Wyoming": "WY",
}
STATE_NAME = {code: name for name, code in STATE_ABBR.items()}


def last_name(name):
    """Last word of a name in lower case, ignoring Jr., Sr., II, III. Used to match names across files."""
    words = [w for w in str(name).strip().split() if w.lower().strip(".") not in {"jr", "sr", "ii", "iii"}]
    return words[-1].lower().strip(".,") if words else ""


def write_table(df, path):
    """Save a table as parquet. If pyarrow is missing, save a .csv next to it and say so."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"  (pyarrow not installed: wrote {csv_path} instead of parquet)")


def read_table(path):
    """Read the parquet file, or the .csv fallback written by write_table."""
    import pandas as pd
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    return pd.read_csv(path.with_suffix(".csv"))


def get_api_key():
    """Return the OpenRouter key from the environment or from a .env file; never print it."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None