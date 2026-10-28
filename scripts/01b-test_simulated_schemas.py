#### Preamble ####
# Purpose: Schema tests of the SIMULATED data with pandera (a data-validation package): columns,
#          types, allowed values and ranges, plus table-level rules (35 races, 100 personas per state,
#          probabilities that add up to 1, one winner per draw). It complements the assert-based
#          checks in scripts/01-test_simulated_data.py; the real data have the same kind of suite in
#          scripts/12b-test_actual_schemas.py.
# Author: Kexin (Coco) Liu
# Date: 27 October 2026
# License: MIT
# Pre-requisites: pip install pandera (in requirements.txt). Run scripts/00-simulate_data.py first.
# Any other information needed? Run from the project root:  python scripts/01b-test_simulated_schemas.py
#   All failures in a table are collected (lazy validation) and printed before the script exits 1.

#### Workspace setup ####
import sys

import pandas as pd

try:
    import pandera.pandas as pa          # pandera 0.24 and later
except ImportError:                       # older pandera
    import pandera as pa

import config

SIM = "data/00-simulated_data/"
STATES = list(config.STATE_ABBR.values())
MODELS = list(config.MODELS)
PARTIES = ["Democratic", "Independent", "Republican"]
CATEGORIES = {
    "age_group": ["18-29", "30-44", "45-64", "65+"],
    "education": ["No HS", "HS grad", "Some college", "2-year", "4-year", "Post-grad"],
    "gender": ["Man", "Woman", "Non-binary", "Other"],
    "race_5": ["White", "Black", "Hispanic", "Asian", "Other"],
    "party_id_7": ["Strong Dem", "Weak Dem", "Lean Dem", "Independent", "Lean Rep", "Weak Rep",
                   "Strong Rep", "Not sure"],
    "ideology": ["Very liberal", "Liberal", "Moderate", "Conservative", "Very conservative", "Not sure"],
}
prob = pa.Check.in_range(0, 1)

#### Data ####
nominees = pd.read_csv(SIM + "simulated_nominees.csv")
candidates = pd.read_csv(SIM + "simulated_candidates.csv")
personas = pd.read_csv(SIM + "simulated_personas.csv")
scores = pd.read_csv(SIM + "simulated_scores.csv", keep_default_na=False, na_values=[""])
forecast = pd.read_parquet(SIM + "simulated_forecast.parquet")
race_states = list(nominees["state"])

#### Schemas ####
schemas = {
    "nominees": (nominees, pa.DataFrameSchema({
        "state": pa.Column(str, pa.Check.isin(STATES), unique=True),
        "race_type": pa.Column(str, pa.Check.isin(["regular", "special"])),
        "left_name": pa.Column(str),
        "left_party": pa.Column(str, pa.Check.isin(["Democratic", "Independent"])),
        "rep_name": pa.Column(str),
        "n_others": pa.Column(int, pa.Check.ge(0)),
    }, checks=pa.Check(lambda d: len(d) == 35, error="there should be 35 races"))),

    "candidates": (candidates, pa.DataFrameSchema({
        "state": pa.Column(str, pa.Check.isin(race_states)),
        "race_type": pa.Column(str, pa.Check.isin(["regular", "special"])),
        "name": pa.Column(str),
        "party": pa.Column(str, pa.Check.isin(PARTIES)),
    }, unique=["state", "name"], checks=pa.Check(
        lambda d: d[d["party"] == "Republican"].groupby("state").size().reindex(race_states).eq(1).all(),
        error="each race should have exactly one Republican"))),

    "personas": (personas, pa.DataFrameSchema({
        "persona_id": pa.Column(str, pa.Check.str_matches(r"^[A-Z]{2}_\d+$"), unique=True),
        "caseid": pa.Column(int, pa.Check.gt(0)),
        "state": pa.Column(str, pa.Check.isin(race_states)),
        **{c: pa.Column(str, pa.Check.isin(v)) for c, v in CATEGORIES.items()},
        "family_income": pa.Column(str),
    }, checks=pa.Check(lambda d: d.groupby("state").size().reindex(race_states).eq(config.N_PER_STATE).all(),
                       error=f"every state should have {config.N_PER_STATE} personas"))),

    "scores": (scores, pa.DataFrameSchema({
        "persona_id": pa.Column(str, pa.Check.isin(list(personas["persona_id"]))),
        "model": pa.Column(str, pa.Check.isin(MODELS)),
        "repeat": pa.Column(int, pa.Check.in_range(1, config.N_REPEATS)),
        "status": pa.Column(str, pa.Check.isin(["ok", "refusal", "empty", "error", "bad_shares"])),
        "p_left": pa.Column(float, prob, nullable=True),
        "p_rep": pa.Column(float, prob, nullable=True),
        "p_other": pa.Column(float, prob, nullable=True),
        "vote_choice": pa.Column(str, pa.Check.isin(PARTIES + ["Undecided"]), nullable=True),
    }, unique=["persona_id", "model", "repeat"], checks=[
        pa.Check(lambda d: ((d["p_left"] + d["p_rep"] + d["p_other"])[d["status"] == "ok"] - 1).abs().lt(1e-6).all(),
                 error="probabilities of a scored answer should add up to 1"),
        pa.Check(lambda d: d.loc[d["status"] != "ok", ["p_left", "p_rep", "p_other"]].isna().all().all(),
                 error="only answers with status ok should have probabilities"),
        pa.Check(lambda d: d.loc[d["status"] == "ok", ["p_left", "p_rep", "p_other"]].notna().all().all(),
                 error="an answer with status ok should not have a missing probability"),
    ])),

    "forecast": (forecast, pa.DataFrameSchema({
        "draw": pa.Column(int, pa.Check.in_range(1, config.N_DRAWS)),
        "state": pa.Column(str, pa.Check.isin(race_states)),
        "model": pa.Column(str, pa.Check.isin(MODELS)),
        "party": pa.Column(str, pa.Check.isin(PARTIES)),
        "share": pa.Column(float, prob),
        "winner": pa.Column(str, pa.Check.isin(PARTIES)),
    }, unique=["draw", "state", "model", "party"], checks=[
        pa.Check(lambda d: (d.groupby(["draw", "state", "model"])["share"].sum() - 1).abs().lt(1e-9).all(),
                 error="the two shares in a race should add up to 1"),
        pa.Check(lambda d: d.groupby(["draw", "state", "model"])["winner"].nunique().eq(1).all(),
                 error="each race and draw should have one winner"),
        pa.Check(lambda d: len(d) == len(race_states) * len(MODELS) * config.N_DRAWS * 2,
                 error="every race, model and draw should have two rows"),
    ])),
}

#### Validate ####
failed = False
for name, (df, schema) in schemas.items():
    try:
        schema.validate(df, lazy=True)
        print(f"{name}: passed")
    except pa.errors.SchemaErrors as err:
        failed = True
        print(f"{name}: FAILED")
        cases = err.failure_cases.assign(check=lambda d: d["check"].astype(str).str.slice(0, 60),
                                         column=lambda d: d["column"].fillna("(table)"))
        print(cases.groupby(["column", "check"]).agg(failures=("failure_case", "size"),
                                                     example=("failure_case", "first")).to_string())
if failed:
    sys.exit(1)
print("\nAll schema tests of the simulated data passed.")
