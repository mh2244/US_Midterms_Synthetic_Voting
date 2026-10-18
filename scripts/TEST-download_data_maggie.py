#### Pre-amble #####
# Maggie's test file - downloading ANES data 
# and cleaning it to get all demographics we will have to examine
####################

# Libraries
# import pandas as pd
# import numpy as np
# 
# # Downloaded 2024 ANES time series data and codebook
# raw.data = pd.read.csv("data/anes_timeseries_2024.csv")

# Codebook - only extract post-election questions pertaining to senate elections
  # We care about how people actually voted, so pre-election results aren't needed
  
# Extract questions related to age, gender, region, education, race, party affiliation

# Once extracted,

"""
Extract Senate-vote and demographic variables from the ANES 2024 Time Series
and save a clean, human-readable file for building crosstab cells.

Usage (from your project root):
    python extract_anes_variables.py
    python extract_anes_variables.py path/to/anes_timeseries_2024.csv

Output: data/anes_senate_demographics.csv
Variable names/codes were verified against the ANES 2024 codebook.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IN_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/anes_timeseries_2024.csv")
OUT_PATH = Path("data/anes_senate_demographics.csv")

# ---------------------------------------------------------------------------
# 1. Raw ANES columns -> friendly names
# ---------------------------------------------------------------------------
COLUMNS = {
    # --- ID & weights ---
    "V240001":  "case_id",
    "V240107a": "weight_pre",    # pre-election full sample (use for pre-only analysis)
    "V240107b": "weight_post",   # post-election full sample (use if post vars involved)
    # --- Demographics ---
    "V241458x": "age",           # age on election day (80 = 80+; -2 = missing)
    "V241550":  "sex_raw",       # 1 Male, 2 Female
    "V241501x": "race_raw",      # race/ethnicity summary (Hispanic folded in)
    "V241465x": "educ_raw",      # 5-category education
    "V241227x": "pid7_raw",      # 7-point party ID
    "V241023":  "state_fips_reg",    # state of registration (FIPS)
    "V243002":  "state_fips_sample", # sample location state (fallback)
    # --- Senate race availability ---
    "V241027":  "senate_race_2024_raw",  # 1 = Senate race in R's registration state
    # --- Senate vote: summary variables (recommended) ---
    "V241077x": "senate_pre_raw",   # pre: vote / intent / preference
    "V242093x": "senate_post_raw",  # post: vote / preference
    "V242098x": "senate_vote_raw",  # pre-post combined: actual vote only
    # --- Senate vote: underlying raw items (for reference) ---
    "V241057":  "pre_voted_senate",
    "V241058":  "pre_vote_senate_who",
    "V241059":  "pre_intend_senate",
    "V241060":  "pre_intend_senate_who",
    "V241061":  "pre_prefer_senate",
    "V241062":  "pre_prefer_senate_who",
}

FIPS_TO_STATE = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE", 11: "DC",
    12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN", 19: "IA", 20: "KS",
    21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA", 26: "MI", 27: "MN", 28: "MS",
    29: "MO", 30: "MT", 31: "NE", 32: "NV", 33: "NH", 34: "NJ", 35: "NM", 36: "NY",
    37: "NC", 38: "ND", 39: "OH", 40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC",
    46: "SD", 47: "TN", 48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV",
    55: "WI", 56: "WY",
}

SEX = {1: "Male", 2: "Female"}
RACE = {
    1: "White", 2: "Black", 3: "Hispanic", 4: "Asian/PI",
    5: "Native/Other", 6: "Multiracial",
}
EDUC = {
    1: "Less than HS", 2: "HS grad", 3: "Some college/assoc.",
    4: "Bachelor's", 5: "Graduate",
}
PID7 = {
    1: "Strong Dem", 2: "Weak Dem", 3: "Lean Dem", 4: "Independent",
    5: "Lean Rep", 6: "Weak Rep", 7: "Strong Rep",
}
PID3 = {1: "Democrat", 2: "Democrat", 3: "Democrat", 4: "Independent",
        5: "Republican", 6: "Republican", 7: "Republican"}

# V241077x / V242093x codes: tens digit = basis, ones digit = party
PARTY_BY_ONES = {0: "Democratic", 1: "Republican", 2: "Other"}
BASIS_BY_TENS = {1: "vote", 2: "intent", 3: "preference"}  # post var: 1=vote, 3=preference
VOTE_FINAL = {1: "Democratic", 2: "Republican", 3: "Other"}  # V242098x


def decode_senate(series: pd.Series):
    """Split ANES summary codes (10/11/12, 20/21/22, 30/31/32) into party + basis."""
    valid = series.where(series >= 10)
    party = (valid % 10).map(PARTY_BY_ONES)
    basis = (valid // 10).map(BASIS_BY_TENS)
    return party, basis


def main():
    df = pd.read_csv(IN_PATH, usecols=list(COLUMNS), low_memory=False)
    df = df.rename(columns=COLUMNS)

    # Some columns load as strings with blank-space entries; coerce to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = pd.DataFrame({"case_id": df["case_id"].astype("Int64")})
    out["weight_pre"] = df["weight_pre"]
    out["weight_post"] = df["weight_post"]  # NaN if no post-election interview

    # --- Demographics: negative codes are refused/DK/missing -> NaN ---
    out["age"] = df["age"].where(df["age"] > 0)
    out["age_group"] = pd.cut(
        out["age"], bins=[17, 29, 44, 64, 120],
        labels=["18-29", "30-44", "45-64", "65+"],
    )
    out["gender"] = df["sex_raw"].map(SEX)
    out["race_ethnicity"] = df["race_raw"].map(RACE)
    out["education"] = df["educ_raw"].map(EDUC)
    out["party_id_7"] = df["pid7_raw"].map(PID7)
    out["party_id_3"] = df["pid7_raw"].map(PID3)  # leaners counted with their party

    # --- State: registration state first, sample state as fallback ---
    fips = df["state_fips_reg"].where(df["state_fips_reg"] > 0)
    fips = fips.fillna(df["state_fips_sample"].where(df["state_fips_sample"] > 0))
    out["state"] = fips.map(FIPS_TO_STATE)

    # --- Senate ---
    out["senate_race_in_state_2024"] = df["senate_race_2024_raw"].map({1: True, 0: False})
    out["senate_pre_party"], out["senate_pre_basis"] = decode_senate(df["senate_pre_raw"])
    out["senate_post_party"], out["senate_post_basis"] = decode_senate(df["senate_post_raw"])
    out["senate_actual_vote"] = df["senate_vote_raw"].map(VOTE_FINAL)

    # Keep the underlying raw items too (raw codes; negatives -> NaN)
    for c in ["pre_voted_senate", "pre_vote_senate_who", "pre_intend_senate",
              "pre_intend_senate_who", "pre_prefer_senate", "pre_prefer_senate_who"]:
        out[c] = df[c].where(df[c] > 0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    # --- Quick sanity report ---
    print(f"Saved {len(out):,} rows x {out.shape[1]} cols -> {OUT_PATH}\n")
    print("Missing share, demographic cells:")
    print(out[["age_group", "gender", "race_ethnicity", "education",
               "party_id_3", "state"]].isna().mean().round(3).to_string(), "\n")
    print("Respondents with a Senate race in their state:",
          int(out["senate_race_in_state_2024"].sum()))
    print("Usable pre-election Senate party (vote/intent/pref):",
          int(out["senate_pre_party"].notna().sum()))
    print("Usable post-election Senate party:", int(out["senate_post_party"].notna().sum()))
    print("Actual-vote Senate (pre+post):", int(out["senate_actual_vote"].notna().sum()))


if __name__ == "__main__":
    main()

