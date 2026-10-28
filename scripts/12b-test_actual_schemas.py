#### Preamble ####
# Purpose: Schema tests of the REAL data with pandera (a data-validation package, the Python
#          counterpart of R's validate/pointblank): every table's columns, types, allowed values
#          and ranges, plus table-level rules (one row per key, shares that add up to 1, 100 personas
#          per state, one winner per draw). It complements the assert-based checks in
#          scripts/12-test_actual_data.py; the simulated data have their own suite in
#          scripts/01b-test_simulated_schemas.py.
# Author: Kexin (Coco) Liu
# Date: 27 October 2026
# License: MIT
# Pre-requisites: pip install pandera (in requirements.txt). Run scripts 06 to 11 for this RUN.
# Any other information needed? Run from the project root:  python scripts/12b-test_actual_schemas.py
#   All failures in a table are collected (lazy validation) and printed before the script exits 1.

#### Workspace setup ####
import sys

import pandas as pd

try:
    import pandera.pandas as pa          # pandera 0.24 and later
except ImportError:                       # older pandera
    import pandera as pa

import config

STATES = list(config.STATE_ABBR.values())
MODELS = list(config.MODELS)
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
nominees = pd.read_csv(config.NOMINEES_FILE)
personas = config.read_table(config.PERSONAS_FILE)
responses = config.read_table(config.RESPONSES_FILE)
scores = config.read_table(config.SCORES_FILE)
forecast = config.read_table(config.FORECAST_FILE)
weights_file = config.RUN_RESULTS / "ivw_weights.csv"
chamber = pd.read_csv(config.RUN_RESULTS / "chamber_summary.csv")
print(f"Schema tests for run {config.RUN}")

#### Schemas ####
schemas = {
    "nominees": (nominees, pa.DataFrameSchema({
        "state": pa.Column(str, pa.Check.isin(STATES), unique=True),
        "left_name": pa.Column(str, pa.Check.str_length(min_value=3)),
        "left_party": pa.Column(str, pa.Check.isin(["Democratic", "Independent"])),
        "rep_name": pa.Column(str, pa.Check.str_length(min_value=3)),
        "race_type": pa.Column(str, pa.Check.isin(["regular", "special"])),
    }, checks=pa.Check(lambda d: len(d) == 35, error="there should be 35 races"))),

    "personas": (personas, pa.DataFrameSchema({
        "persona_id": pa.Column(str, pa.Check.str_matches(r"^[A-Z]{2}_\d+$"), unique=True),
        "state": pa.Column(str, pa.Check.isin(list(nominees["state"]))),
        **{c: pa.Column(str, pa.Check.isin(v)) for c, v in CATEGORIES.items()},
        "persona_text": pa.Column(str, pa.Check.str_length(min_value=50)),
        "n_pool": pa.Column(int, pa.Check.gt(0)),
        "n_eff": pa.Column(float, pa.Check.gt(0)),
    }, checks=[pa.Check(lambda d: d.groupby("state").size().eq(config.N_PER_STATE).all(),
                        error=f"every state should have {config.N_PER_STATE} personas"),
               pa.Check(lambda d: d["state"].nunique() == len(nominees), error="a race state has no personas")])),

    "responses": (responses, pa.DataFrameSchema({
        "prompt_id": pa.Column(str),
        "run": pa.Column(str, pa.Check.eq(config.RUN)),
        "persona_id": pa.Column(str, pa.Check.isin(list(personas["persona_id"]))),
        "model": pa.Column(str, pa.Check.isin(MODELS)),
        "repeat": pa.Column(int, pa.Check.in_range(1, config.N_REPEATS)),
        "input_tokens": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
        "output_tokens": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
        "queried_at_utc": pa.Column(str, pa.Check.str_matches(r"^\d{4}-\d{2}-\d{2}T")),
    }, unique=["prompt_id", "model", "repeat"], checks=pa.Check(
        lambda d: len(d) == len(personas) * len(MODELS) * config.N_REPEATS,
        error="every persona should be asked by every model, N_REPEATS times"))),

    "scores": (scores, pa.DataFrameSchema({
        "model": pa.Column(str, pa.Check.isin(MODELS)),
        "status": pa.Column(str, pa.Check.isin(["ok", "refusal", "empty", "error", "bad_shares"])),
        "p_left": pa.Column(float, prob, nullable=True),
        "p_rep": pa.Column(float, prob, nullable=True),
        "p_other": pa.Column(float, prob, nullable=True),
    }, checks=[
        pa.Check(lambda d: ((d["p_left"] + d["p_rep"] + d["p_other"])[d["status"] == "ok"] - 1).abs().lt(1e-6).all(),
                 error="probabilities of a scored answer should add up to 1"),
        pa.Check(lambda d: d.loc[d["status"] != "ok", "p_left"].isna().all(),
                 error="only answers with status ok should have probabilities"),
    ])),

    "forecast": (forecast, pa.DataFrameSchema({
        "draw": pa.Column(int, pa.Check.ge(1)),
        "state": pa.Column(str, pa.Check.isin(list(nominees["state"]))),
        "model": pa.Column(str, pa.Check.isin(MODELS)),
        "party": pa.Column(str, pa.Check.isin(["Democratic", "Independent", "Republican"])),
        "share": pa.Column(float, prob),
        "winner": pa.Column(str, pa.Check.isin(["Democratic", "Independent", "Republican"])),
    }, unique=["draw", "state", "model", "party"], checks=[
        pa.Check(lambda d: (d.groupby(["draw", "state", "model"])["share"].sum() - 1).abs().lt(1e-9).all(),
                 error="the two shares in a race should add up to 1"),
        pa.Check(lambda d: d.groupby(["draw", "state", "model"])["winner"].nunique().eq(1).all(),
                 error="each race and draw should have one winner"),
    ])),

    "chamber summary": (chamber, pa.DataFrameSchema({
        "model": pa.Column(str, pa.Check.isin(MODELS), unique=True),
        "dem_median": pa.Column(float, pa.Check.in_range(0, 100), coerce=True),
        "rep_median": pa.Column(float, pa.Check.in_range(0, 100), coerce=True),
        "p_rep_control": pa.Column(float, prob),
        "p_dem_51_without_ind": pa.Column(float, prob),
    }, checks=pa.Check(lambda d: (d["p_rep_control"] + d["p_dem_51_without_ind"]).le(1 + 1e-9).all(),
                       error="Republican control and 51 Democratic seats cannot both happen"))),
}
if weights_file.exists():
    schemas["inverse-variance weights"] = (pd.read_csv(weights_file), pa.DataFrameSchema({
        "model": pa.Column(str, pa.Check.isin(MODELS)),
        "state": pa.Column(str, pa.Check.isin(list(nominees["state"]))),
        "source": pa.Column(str, pa.Check.isin(["poll", "2024 + swing"])),
        "w_synthetic": pa.Column(float, prob),
        "combined": pa.Column(float, prob),
        "var_m": pa.Column(float, pa.Check.gt(0)),
        "var_t": pa.Column(float, pa.Check.gt(0)),
    }, unique=["model", "state"]))

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
print("\nAll schema tests passed.")
