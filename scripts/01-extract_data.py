"""
Pre-amble:

Purpose: Extract the relevant variables from the 2024 CES dataset.
The variables we want are senate-vote variables and demographic variables.
These will be used to build cross-tabs later on.
Authors: Maggie, Coco, Amelia
Date: November 2nd, 2026
Contact: maggieh.huang@mail.utoronto.ca
License: MIT
Pre-requisites: Run 00-download_data.py first and ensure that the 2024 CES dataset is saved
under the "data" subfolder. 
Other Information: In the 2026 Senate race, the following states will have races, according to https://ballotpedia.org/U.S._Senate_battlegrounds,_2026

1. Alabama
2. Alaska
3. Arkansas
4. Colorado
5. Delaware
6. Florida
7. Georgia
8. Idaho
9. Illinois
10. Iowa
11. Kansas
12. Kentucky
13. Louisiana
14. Maine
15. Massachusetts
16. Michigan
17. Minnesota
18. Mississippi
19. Montana
20. Nebraska
21. New Hampshire
22. New Jersey
23. New Mexico
24. North Carolina
25. Ohio
26. Oklahoma
27. Oregon
28. Rhode Island
29. South Carolina
30. South Dakota
31. Tennessee
32. Texas
33. Virginia
34. West Virginia
35. Wyoming
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_IN = Path("data/CCES24_Common_OUTPUT_vv_topost_final.csv")
OUT_PATH = Path("data/ces2024_senate_demographics.csv")

# Columns named in the CES 2024 guide
DEMOS = ["birthyr", # year respondent was born in
         "gender4", # options: man/woman/non-binary/other
         "educ",  # options: No HS/HS grad/Some college/2-year/4-year/post-grad
         "race", # options: white/black/hispanic/asian/native american/middle eastern/2+ races/other
         "hispanic", #yes/no, if they are of spanish/latino/hispanic origin or descent
         "pid3", # 3 point party ID: democrat/republican/independent/other/unsure
         "pid7", # 7 point party ID: strong democrat/not very strong democrat/strong republican/not very strong republican/
         #lean democrat/lean republican/independent/not sure/don't know
         "inputstate", # state of residence 
         "region", # census region: northwest/midwest/south/west
         "votereg", # voting registration status: yes/no/don't know
         "votereg_post", # voting registration status post: yes/no/don't know 
         "cit1", # US citizenship: yes/no
         "ideo5", # Ideology: very liberal/liberal/moderate/conservative/very conservative/not sure
         "faminc_new" # Family income
         ]

# 2024 CES Common Content includes weights for both pre-election and post-election waves 
# of the study
# Weights are constructed such that sample is representative of different populations
# Either all adult Americans or adult Americans registered to vote
# The codebook gives the following recommendations:
    # Use commonweight if research is about characterizing behaviours/opinions of all Americans
    # Use commonpostweight if using ANY items from post-election wave of questionnaire
    # vvweight if want to characterize opinions/behaviours of voters/registered voters
    #   If using vv, then include vote validation variables (votereg, votereg_post) 
WEIGHTS = ["commonweight", # all respondents (all adults)
           "commonpostweight", # answered both waves (all adults)
           "vvweight", # matched to validated registration record (all registered)
           "vvweight_post" # answered both waves and matched to registration record (all registered)
           ]

VALIDATION = ["TS_g2024", # how someone voted in 2024 general election (if missing, respondent did not have voting record)
              "CC24_401" # question asking for if they voted in the 2024 general election: did not vote/thought about voting but didn't
              # usually vote but not this time/attempted to but couldn't/voted
              ]

# Candidate slots (1/2/3) are piped per state, so party comes from these fields,
# IF they are present in the data file. --inspect tells you.
SLOT_PARTY_GUESS = ["SenCand1Party", "SenCand2Party", "SenCand3Party",
                    "SenCand1Party2", "SenCand2Party2"]


SENATE = [
    "CC24_365_voted",   # pre-wave: Senate vote (already voted early)
    "CC24_365",         # pre-wave: Senate preference
    "CC24_411",         # post-wave: Senate vote <--  Main outcome we care about
    "CC24_365b_voted", "CC24_365b",   # pre-wave special-election race (Nebraska)
    "CC24_411b",                      # post-wave special-election race
    "CC24_410",         # post-wave presidential vote (useful cross-check)
]

FIPS_TO_STATE = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE", 11: "DC",
    12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN", 19: "IA", 20: "KS",
    21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA", 26: "MI", 27: "MN", 28: "MS",
    29: "MO", 30: "MT", 31: "NE", 32: "NV", 33: "NH", 34: "NJ", 35: "NM", 36: "NY",
    37: "NC", 38: "ND", 39: "OH", 40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC",
    46: "SD", 47: "TN", 48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV",
    55: "WI", 56: "WY",
}

# To get the code and labels, look at pre- and post- questionnaires
EDUC = {1: "No HS", 2: "HS grad", 3: "Some college", 4: "2-year", 5: "4-year", 6: "Post-grad"}
RACE = {1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "Native American",
        6: "Two or more", 7: "Other", 8: "Middle Eastern"}  # numeric order: confirm in --inspect
GENDER4 = {1: "Man", 2: "Woman", 3: "Non-binary", 4: "Other"}
PID3 = {1: "Democrat", 2: "Republican", 3: "Independent", 4: "Other", 5: "Not sure"}
PID7 = {1: "Strong Dem", 2: "Weak Dem", 3: "Lean Dem", 4: "Independent",
        5: "Lean Rep", 6: "Weak Rep", 7: "Strong Rep", 8: "Not sure"}
IDEO5 = {1: "Very liberal", 2: "Liberal", 3: "Moderate",
         4: "Conservative", 5: "Very conservative", 6: "Not sure"}
PRES24 = {1: "Harris", 2: "Trump", 3: "Kennedy", 4: "Stein", 5: "West",
          6: "Oliver", 8: "Other", 9: "Did not vote for President"}
FAMINC = {
    1: "Less than $10,000",
    2: "$10,000 - $19,999",
    3: "$20,000 - $29,999",
    4: "$30,000 - $39,999",
    5: "$40,000 - $49,999",
    6: "$50,000 - $59,999",
    7: "$60,000 - $69,999",
    8: "$70,000 - $79,999",
    9: "$80,000 - $99,999",
    10: "$100,000 - $119,999",
    11: "$120,000 - $149,999",
    12: "$150,000 - $199,999",
    13: "$200,000 - $249,999",
    14: "$250,000 - $349,999",
    15: "$350,000 - $499,999",
    16: "$500,000 or more",
    97: "Prefer not to say"
}

# Define functions

def load(path):
    return pd.read_csv(path, low_memory=False)

def inspect(df):
    want = DEMOS + WEIGHTS + VALIDATION + SENATE + SLOT_PARTY_GUESS
    print(f"{len(df):,} rows x {df.shape[1]} columns\n")
    print("Expected columns present?")
    for c in want:
        print(f"  {'OK     ' if c in df.columns else 'MISSING'} {c}")
    print("\nColumns containing 'sen' (case-insens.) - look for candidate/party fields:")
    print("  ", [c for c in df.columns if "sen" in c.lower()])
    print("\nValue counts for key variables (to confirm how they are coded):")
    for c in ["race", "educ", "pid3", "pid7", "gender4", "CC24_411", "CC24_365"]:
        if c in df.columns:
            print(f"\n{c}:\n{df[c].value_counts(dropna=False).head(12).to_string()}")

def decode(series, mapping):
    """Works whether the column holds numeric codes or already-text labels."""
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() > 0.5:          # numeric coding
        return num.map(mapping)
    return series.astype("string").str.strip()   # already labels


PLACEHOLDERS = {"-1", "__NA__", "", "nan", "NaN"}
# States with a 2024 U.S. Senate race (FIPS). Nebraska had two (regular + special).
SENATE_2024_FIPS = {4, 6, 9, 10, 12, 15, 18, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34,
                    35, 36, 38, 39, 42, 44, 47, 48, 49, 50, 51, 53, 54, 55, 56}

def clean_party(series):
    """CES party-label fields contain placeholder strings ('-1', '__NA__'); make them NaN."""
    s = series.astype("string").str.strip()
    return s.mask(s.isin(PLACEHOLDERS))


def slot_to_party(df, vote_col, party_cols_primary, party_cols_fallback, other_code):
    """Candidate slot answers (1/2/3) -> that respondent's candidate party label.
    Uses the first set of party fields, then the second (they agree on all but ~5 rows).
    other_code = the code for the write-in 'Other' option (varies by question; checked
    against the guide: CC24_411 -> 5, CC24_365_voted -> 7, CC24_365 -> 7)."""

    # Force conversion
    v = pd.to_numeric(df[vote_col], errors="coerce")
    out = pd.Series(pd.NA, index=df.index, dtype="string")
    out[v == other_code] = "Other"
    for slot, pc1, pc2 in zip([1, 2, 3], party_cols_primary, party_cols_fallback):
        mask = v == slot
        lab = clean_party(df[pc1]) if pc1 in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")
        if pc2 in df.columns:
            lab = lab.fillna(clean_party(df[pc2]))
        out[mask] = lab[mask]
    return out

def three_way(party):
    """Democratic / Republican / Other (Independent, Libertarian, Green, write-in...)."""
    out = party.where(party.isin(["Democratic", "Republican"]), "Other")
    return out.where(party.notna())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(DEFAULT_IN))
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    df = load(a.path)

    if a.inspect:
        inspect(df)
        return

    out = pd.DataFrame(index=df.index)
    if "caseid" in df.columns:
        out["caseid"] = df["caseid"]
    for w in WEIGHTS:
        out[w] = pd.to_numeric(df[w], errors="coerce")

    # ---------- demographics (raw CES categories) ----------
    out["age"] = 2024 - pd.to_numeric(df["birthyr"], errors="coerce")
    out["age_group"] = pd.cut(out["age"], [17, 29, 44, 64, 130],
                              labels=["18-29", "30-44", "45-64", "65+"])
    out["gender"] = decode(df["gender4"], GENDER4)
    out["education"] = decode(df["educ"], EDUC)
    race = decode(df["race"], RACE)
    hisp = pd.to_numeric(df["hispanic"], errors="coerce") == 1
    # CES guide rule: Hispanic if race==Hispanic OR Hispanic-origin question == Yes
    out["race_ethnicity"] = np.where(hisp | (race == "Hispanic"), "Hispanic", race)
    out["party_id_3"] = decode(df["pid3"], PID3)
    out["party_id_7"] = decode(df["pid7"], PID7)
    fips = pd.to_numeric(df["inputstate"], errors="coerce")
    out["state_fips"] = fips
    out["state"] = fips.map(FIPS_TO_STATE)
    out["region"] = df["region"]

    # ----------- ideology demographics ------------------
    out["ideology"] = decode(df["ideo5"], IDEO5)

    # ---------- collapsed versions for matching Census/ACS later ----------
    out["education_4"] = out["education"].map({
        "No HS": "HS or less", "HS grad": "HS or less",
        "Some college": "Some college/2-yr", "2-year": "Some college/2-yr",
        "4-year": "Bachelor's", "Post-grad": "Postgraduate"})
    out["race_5"] = out["race_ethnicity"].where(
        out["race_ethnicity"].isin(["White", "Black", "Hispanic", "Asian"]), "Other")
    out.loc[out["race_ethnicity"].isna(), "race_5"] = np.nan

    # ---------- turnout / validation ----------
    ts = pd.to_numeric(df["TS_g2024"], errors="coerce")
    out["TS_g2024"] = ts
    out["validated_voter"] = ts.between(1, 6)      # guide: any value below 7 = validated vote
    out["CC24_401"] = df["CC24_401"]
    out["votereg"] = df["votereg"]
    out["cit1"] = df["cit1"]

    # ---------- Senate ----------
    post_cols = ["SenCand1Party_post", "SenCand2Party_post", "SenCand3Party_post"]
    pre_cols = ["SenCand1Party", "SenCand2Party", "SenCand3Party"]
    out["senate_race_in_state_2024"] = fips.isin(SENATE_2024_FIPS)

    for c in ["CC24_411", "CC24_365_voted", "CC24_365", "CC24_411b", "CC24_365b_voted", "CC24_365b"]:
        out[c] = df[c]                       # raw codes kept for reference

    post_party = slot_to_party(df, "CC24_411", post_cols, pre_cols, other_code=5)
    prevoted_party = slot_to_party(df, "CC24_365_voted", pre_cols, post_cols, other_code=7)
    pref_party = slot_to_party(df, "CC24_365", pre_cols, post_cols, other_code=7)
    out["senate_party_post_raw"] = post_party       # keeps 'Independent', 'Libertarian', ...
    out["senate_vote_post"] = three_way(post_party)           # MAIN OUTCOME: D / R / Other
    out["senate_vote_prevoted"] = three_way(prevoted_party)
    out["senate_pref_pre"] = three_way(pref_party)

    # "best available": actual post-election vote, else early-vote answer, else pre-election preference
    best = out["senate_vote_post"].copy()
    src = pd.Series(pd.NA, index=df.index, dtype="string").mask(best.notna(), "post vote")
    for name, col in [("pre-wave early vote", "senate_vote_prevoted"), ("pre-wave preference", "senate_pref_pre")]:
        fill = best.isna() & out[col].notna()
        best = best.mask(fill, out[col])
        src = src.mask(fill, name)
    out["senate_vote_best"] = best
    out["senate_vote_best_source"] = src

# ------------ Income -----------------------
    out["faminc_raw"] = pd.to_numeric(df["faminc_new"], errors="coerce")
    out["family_income"] = decode(df["faminc_new"], FAMINC)
    
    # Cleaned version mapping refusal/skipped codes (97, 998, 999) to NaN for analysis
    out["family_income_clean"] = out["faminc_raw"].replace([97, 998, 999], np.nan)
    
     # ---------- 2024 presidential vote (post-election question) ----------
    out["CC24_410"] = df["CC24_410"]                       # raw code, kept for reference
    out["pres_vote_2024"] = decode(df["CC24_410"], PRES24)  # readable name
    voted_pres = out["pres_vote_2024"].notna() & (out["pres_vote_2024"] != "Did not vote for President")
    out["pres_vote_3way"] = (out["pres_vote_2024"]
                             .map({"Harris": "Democratic", "Trump": "Republican"})
                             .fillna("Other")
                             .where(voted_pres))


    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    # ---------- sanity report ----------
    print(f"Saved {len(out):,} rows x {out.shape[1]} cols -> {OUT_PATH}\n")
    print("Senate vote, post-election (unweighted counts):")
    print(out["senate_vote_post"].value_counts(dropna=False).to_string(), "\n")
    m = out["validated_voter"] & out["senate_vote_post"].notna() & out["vvweight_post"].notna()
    print(f"Usable for crosstabs (validated voter + post Senate vote + vvweight_post): {int(m.sum()):,}")
    w = out.loc[m, "vvweight_post"]
    print("Weighted Senate vote share among those (D/R/Other):")
    print((w.groupby(out.loc[m, "senate_vote_post"]).sum() / w.sum()).round(3).to_string())
    print("\nWeighted Senate vote by 3-way party ID (rows sum to 1):")
    t = out.loc[m].pivot_table(index="party_id_3", columns="senate_vote_post",
                               values="vvweight_post", aggfunc="sum").fillna(0)
    print(t.div(t.sum(axis=1), axis=0).round(2).to_string())
    print("\nMissing share in demographic cells:")
    print(out[["age_group", "gender", "race_ethnicity", "education", "party_id_3", "state"]]
          .isna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
