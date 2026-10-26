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
from pathlib import Path

#### Settings you are likely to change ####
SEED = 2026               # one seed for every random step, so reruns match
N_PER_STATE = 100           # personas per state. Test run = 5; the real run was planned at 100
N_REPEATS = 3             # times each prompt is sent to each model (planned: 3)
N_DRAWS = 1000            # simulated outcomes per state and model (same as the simulated data)

# Short names are used in the data; the values are the OpenRouter model ids.
# CHECK both ids on https://openrouter.ai/models before running; stealth models can be renamed
# or withdrawn without notice, which is why the id the API returns is saved with every answer.
MODELS = {
    "laguna": "poolside/laguna-s-2.1:free", #Laguna will not be available Oct.31st
    "nemotron": "nvidia/nemotron-3-ultra-550b-a55b:free",
}

# Embedding model for SSR (script 10). "local" needs: pip install sentence-transformers
EMBEDDER = "local"
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

# Date the news headlines and polls were collected (write it in the paper)
AS_OF = "2026-10-21"

#### Paths (relative to the project root) ####
DATA = Path("data")
ANALYSIS = DATA / "02-analysis_data"
RESULTS = DATA / "03-results"
CES_FILE = DATA / "ces2024_senate_demographics.csv"
CANDIDATES_RAW = DATA / "senate_candidates_2026.csv"
NEWS_FILE = DATA / "news_google" / "news_google.csv"
POLLS_FILE = DATA / "polling_data" / "02-analysis_data" / "senate_polls_2026_general.csv"

NOMINEES_FILE = ANALYSIS / "senate_nominees_2026.csv"
CANDIDATES_CLEAN = ANALYSIS / "candidates_clean.csv"
PERSONAS_FILE = ANALYSIS / "personas.parquet"
PROMPTS_FILE = ANALYSIS / "prompts.parquet"
RESPONSES_FILE = ANALYSIS / "responses.parquet"
SCORES_FILE = ANALYSIS / "scores.parquet"
FORECAST_FILE = RESULTS / "forecast.parquet"

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