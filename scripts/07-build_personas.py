#### Preamble ####
# Purpose: Builds the personas ("digital twins"): for each 2026 race state, draws real CES 2024
#          respondents with probability proportional to their survey weight, and writes a
#          short text description of each one.
# Author: Maggie, Coco, Amelia
# Date: 22 October 2026
# License: MIT
# Pre-requisites: Run scripts/03_extract_data.py and scripts/06-clean_candidates.py first.
# Any other information needed? Run from the project root. Set N_PER_STATE in config.py
#   (5 for the test run). Add --n 100 for the real run.
#
# WHY draws proportional to weight: the CES weights already make each state's sample match its
# registered voters (CES guide, p. 17), so a weighted draw gives a persona sample that mirrors
# the state without any further weighting. This is the "digital twin" approach used by
# Murray & Alexander (2026) and Argyle et al. (2023, Sec. 4).
# WHY with replacement, for EVERY state: some states have fewer respondents than the 100
# personas planned (Wyoming ~54 with vvweight_post), so repeated draws are needed there. Using
# the same rule in all states keeps the method identical and keeps the draw proportional to the
# weights; drawing without replacement would not preserve them.
# WHY the 2024 presidential vote is NOT in the persona: it nearly gives the answer away once
# party and ideology are known, and we keep it as a held-out check instead.
# WHY validated voters only (decided 2026-10-26): vvweight_post covers every respondent matched to
# a registration record, including registered people who did NOT vote in 2024 (about 8% of the
# weight). We forecast votes, so the pool keeps only respondents whose 2024 vote was confirmed
# in the voter file (validated_voter, from TS_g2024 codes 1-6 in script 03).
# LIMIT: 2024 voters stand in for the 2026 electorate; midterm turnout is lower and differs.


#### Workspace setup ####
import argparse

import numpy as np
import pandas as pd

import config

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=config.N_PER_STATE, help="personas per state")
args = ap.parse_args()

# The 7 persona variables (same names and labels as the simulated personas)
VARS = ["age_group", "family_income", "education", "gender", "race_5", "party_id_7", "ideology"]

EDUCATION_TEXT = {
    "No HS": "You did not finish high school.",
    "HS grad": "Your highest level of education is a high school diploma.",
    "Some college": "You have some college education but no degree.",
    "2-year": "You have a two-year college degree.",
    "4-year": "You have a four-year college degree.",
    "Post-grad": "You have a postgraduate degree.",
}
PARTY_TEXT = {
    "Strong Dem": "You are a strong Democrat.",
    "Weak Dem": "You are a Democrat, but not a very strong one.",
    "Lean Dem": "You are an independent who leans toward the Democratic Party.",
    "Independent": "You are a political independent.",
    "Lean Rep": "You are an independent who leans toward the Republican Party.",
    "Weak Rep": "You are a Republican, but not a very strong one.",
    "Strong Rep": "You are a strong Republican.",
    "Not sure": "You are not sure how you identify politically.",
}
IDEOLOGY_TEXT = {
    "Very liberal": "Your political views are very liberal.",
    "Liberal": "Your political views are liberal.",
    "Moderate": "Your political views are moderate.",
    "Conservative": "Your political views are conservative.",
    "Very conservative": "Your political views are very conservative.",
    "Not sure": "You are not sure how to describe your political views.",
}
RACE_TEXT = {"White": "white", "Black": "Black", "Hispanic": "Hispanic", "Asian": "Asian",
             "Other": "mixed-race or other-race"}
GENDER_TEXT = {"Man": "man", "Woman": "woman", "Non-binary": "non-binary person", "Other": "person"}
AGE_TEXT = {"18-29": "aged 18 to 29", "30-44": "aged 30 to 44", "45-64": "aged 45 to 64",
            "65+": "aged 65 or older"}


def income_text(bracket):
    """'$30,000 - $39,999' -> 'Your household income is between $30,000 and $39,999.'"""
    if bracket == "Prefer not to say":
        return ""
    if " - " in bracket:
        return "Your household income is between " + bracket.replace(" - ", " and ") + "."
    return "Your household income is " + bracket[0].lower() + bracket[1:] + "."


def persona_text(row):
    """One short description. Does NOT mention the 2024 presidential vote."""
    race = RACE_TEXT[row["race_5"]]
    article = "an" if race[0] in "aeiouAEIOU" else "a"
    parts = [
        f"You are {article} {race} {GENDER_TEXT[row['gender']]}, {AGE_TEXT[row['age_group']]}, "
        f"living in {config.STATE_NAME[row['state']]}.",
        EDUCATION_TEXT[row["education"]],
        income_text(row["family_income"]),
        PARTY_TEXT[row["party_id_7"]],
        IDEOLOGY_TEXT[row["ideology"]],
    ]
    return " ".join(p for p in parts if p)


def kish_neff(w):
    """Effective sample size (sum of weights)^2 / sum of squared weights."""
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum())


#### Build the pool ####
nominees = pd.read_csv(config.NOMINEES_FILE)
states = list(nominees["state"])

ces = pd.read_csv(config.CES_FILE, low_memory=False)
pool = ces[ces["state"].isin(states) & ces["vvweight_post"].notna() & (ces["vvweight_post"] > 0)].copy()
n_registered = len(pool)
pool = pool[pool["validated_voter"].astype(str).str.lower() == "true"].copy()   # 2024 validated voters only
n_start = len(pool)
print(f"Pool: kept {n_start:,} validated 2024 voters of {n_registered:,} registered respondents "
      f"({n_registered - n_start:,} non-voters dropped)")

# Keep "Not sure" and "Prefer not to say" as categories instead of dropping those people
pool["party_id_7"] = pool["party_id_7"].fillna("Not sure")
pool["ideology"] = pool["ideology"].fillna("Not sure")
pool["family_income"] = pool["family_income"].fillna("Prefer not to say")

# Drop the few people missing something we cannot sensibly fill in
pool = pool.dropna(subset=["age_group", "education", "gender", "race_5"])
print(f"Pool: {n_start:,} validated voters with vvweight_post in {len(states)} states; "
      f"{n_start - len(pool)} dropped for missing age/education/gender/race")

#### Draw personas ####
rng = np.random.default_rng(config.SEED)
drawn, pool_info = [], []

for state in states:
    sub = pool[pool["state"] == state]
    p = sub["vvweight_post"].to_numpy()
    p = p / p.sum()
    idx = rng.choice(len(sub), size=args.n, replace=True, p=p)
    pick = sub.iloc[idx][["caseid", "state"] + VARS].copy()
    pick["persona_id"] = [f"{state}_{i + 1}" for i in range(args.n)]
    drawn.append(pick)
    pool_info.append({"state": state, "n_pool": len(sub), "n_eff": round(kish_neff(p), 1)})

personas = pd.concat(drawn, ignore_index=True)
personas["persona_text"] = personas.apply(persona_text, axis=1)
personas = personas.merge(pd.DataFrame(pool_info), on="state")
# True when this CES respondent was already drawn earlier for the same state (a repeated draw)
personas["repeat_draw"] = personas.duplicated(["state", "caseid"])
personas = personas[["persona_id", "caseid", "state"] + VARS + ["persona_text", "n_pool", "n_eff", "repeat_draw"]]

#### Checks and save ####
assert personas["persona_id"].is_unique
assert (personas.groupby("state").size() == args.n).all()
assert not personas["persona_text"].str.contains("Trump|Harris|president", case=False).any()

config.write_table(personas, config.PERSONAS_FILE)
personas.to_csv(config.ANALYSIS / "personas.csv", index=False)   # easy to open in Excel

print(f"\nSaved {len(personas)} personas ({args.n} per state) to {config.PERSONAS_FILE}")
print("States with the thinnest pools (small n_eff means the forecast is weaker there):")
print(pd.DataFrame(pool_info).sort_values("n_eff").head(6).to_string(index=False))
print("\nExample persona:\n" + personas["persona_text"].iloc[0])
print("Repeated draws (same CES person drawn again in a state):", int(personas["repeat_draw"].sum()))
print("Most repeats by state:")
print(personas.groupby("state")["repeat_draw"].sum().sort_values(ascending=False).head(5).to_string())
