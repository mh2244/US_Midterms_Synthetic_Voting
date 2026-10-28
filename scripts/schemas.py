#### Preamble ####
# Purpose: The rules every data table must follow, written once as pandera schemas (pandera is
#          a data-validation package, the Python counterpart of R's validate and pointblank) and
#          shared by the two test suites: scripts/01-test_simulated_data.py (simulated data) and
#          scripts/12-test_actual_data.py (real data). Keeping the rules in one place means the
#          simulated and the real data are held to exactly the same standard.
# Author: Kexin (Coco) Liu
# Date: 27 October 2026
# License: MIT
# Pre-requisites: pip install pandera (in requirements.txt).
# Any other information needed? This file only defines schemas; it does not read any data.
#   A schema checks each column's type, allowed values and range, uniqueness of the key columns,
#   and table-level rules (for example, that probabilities add up to 1).

#### Workspace setup ####
try:
    import pandera.pandas as pa          # pandera 0.24 and later
except ImportError:                       # older pandera
    import pandera as pa

#### Allowed values ####
PARTIES = ["Democratic", "Independent", "Republican"]
LEFT_PARTIES = ["Democratic", "Independent"]
RACE_TYPES = ["regular", "special"]
STATUSES = ["ok", "refusal", "empty", "error", "bad_shares"]
VOTE_CHOICES = PARTIES + ["Undecided"]

# The seven persona variables and their labels (the same as scripts 00 and 07)
PERSONA_LABELS = {
    "age_group": ["18-29", "30-44", "45-64", "65+"],
    "family_income": [
        "Less than $10,000", "$10,000 - $19,999", "$20,000 - $29,999",
        "$30,000 - $39,999", "$40,000 - $49,999", "$50,000 - $59,999",
        "$60,000 - $69,999", "$70,000 - $79,999", "$80,000 - $99,999",
        "$100,000 - $119,999", "$120,000 - $149,999", "$150,000 - $199,999",
        "$200,000 - $249,999", "$250,000 - $349,999", "$350,000 - $499,999",
        "$500,000 or more", "Prefer not to say"],
    "education": ["No HS", "HS grad", "Some college", "2-year", "4-year", "Post-grad"],
    "gender": ["Man", "Woman", "Non-binary", "Other"],
    "race_5": ["White", "Black", "Hispanic", "Asian", "Other"],
    "party_id_7": ["Strong Dem", "Weak Dem", "Lean Dem", "Independent",
                   "Lean Rep", "Weak Rep", "Strong Rep", "Not sure"],
    "ideology": ["Very liberal", "Liberal", "Moderate", "Conservative",
                 "Very conservative", "Not sure"],
}
P_COLS = ["p_left", "p_rep", "p_other"]
TOL = 1e-6                                # rounding tolerance when probabilities are added up

prob = pa.Check.in_range(0, 1)


#### Schemas ####
def nominees_schema(states, n_races=35):
    """One row per race: a known state, two named candidates, a left party and a race type."""
    return pa.DataFrameSchema({
        "state": pa.Column(str, pa.Check.isin(states), unique=True),
        "race_type": pa.Column(str, pa.Check.isin(RACE_TYPES)),
        "left_name": pa.Column(str, pa.Check.str_length(min_value=3)),
        "left_party": pa.Column(str, pa.Check.isin(LEFT_PARTIES)),
        "rep_name": pa.Column(str, pa.Check.str_length(min_value=3)),
        "n_others": pa.Column(int, pa.Check.ge(0)),
    }, checks=[
        pa.Check(lambda d: len(d) == n_races, error=f"there should be {n_races} races"),
        pa.Check(lambda d: (d["race_type"] == "special").sum() == 2, error="there should be 2 special elections"),
        pa.Check(lambda d: (d["left_name"] != d["rep_name"]).all(), error="the two candidates have the same name"),
    ])


def candidates_schema(states):
    """Long candidate file: one row per candidate, each race with exactly one Republican."""
    return pa.DataFrameSchema({
        "state": pa.Column(str, pa.Check.isin(states)),
        "race_type": pa.Column(str, pa.Check.isin(RACE_TYPES)),
        "name": pa.Column(str, pa.Check.str_length(min_value=3)),
        "party": pa.Column(str, pa.Check.isin(PARTIES)),
    }, unique=["state", "name"], checks=pa.Check(
        lambda d: d[d["party"] == "Republican"].groupby("state").size().reindex(states).eq(1).all(),
        error="each race should have exactly one Republican"))


def personas_schema(states, n_per_state, with_text=False):
    """One row per persona, the same number in every state, only allowed labels, no missing values.
    with_text adds the columns that only the real personas have (persona_text, n_pool, n_eff)."""
    columns = {
        "persona_id": pa.Column(str, pa.Check.str_matches(r"^[A-Z]{2}_\d+$"), unique=True),
        "caseid": pa.Column(int, pa.Check.gt(0)),
        "state": pa.Column(str, pa.Check.isin(states)),
        **{c: pa.Column(str, pa.Check.isin(v)) for c, v in PERSONA_LABELS.items()},
    }
    if with_text:
        columns.update({
            "persona_text": pa.Column(str, pa.Check.str_length(min_value=50)),
            "n_pool": pa.Column(int, pa.Check.gt(0)),
            "n_eff": pa.Column(float, pa.Check.gt(0)),
        })
    return pa.DataFrameSchema(columns, checks=[
        pa.Check(lambda d: d.groupby("state").size().reindex(states).eq(n_per_state).all(),
                 error=f"every state should have {n_per_state} personas"),
        pa.Check(lambda d: "vvweight_post" not in d.columns, error="the CES weight should be dropped after sampling"),
    ])


def scores_schema(models, n_repeats, persona_ids):
    """One scored answer per persona, model and repeat. Only answers with status "ok" have
    probabilities; those are between 0 and 1 and add up to 1. This is what keeps a model that did
    not answer (an NA) out of the forecast."""
    return pa.DataFrameSchema({
        "persona_id": pa.Column(str, pa.Check.isin(persona_ids)),
        "model": pa.Column(str, pa.Check.isin(models)),
        "repeat": pa.Column(int, pa.Check.in_range(1, n_repeats)),
        "status": pa.Column(str, pa.Check.isin(STATUSES)),
        **{c: pa.Column(float, prob, nullable=True) for c in P_COLS},
    }, unique=["persona_id", "model", "repeat"], checks=[
        pa.Check(lambda d: (d.loc[d["status"] == "ok", P_COLS].sum(axis=1) - 1).abs().lt(TOL).all(),
                 error="probabilities of a scored answer should add up to 1"),
        pa.Check(lambda d: d.loc[d["status"] == "ok", P_COLS].notna().all().all(),
                 error="an answer with status ok is missing a probability"),
        pa.Check(lambda d: d.loc[d["status"] != "ok", P_COLS].isna().all().all(),
                 error="an answer that is not ok has a probability"),
    ])


def forecast_schema(states, models, n_draws=None):
    """One row per draw, race, model and party, with a share between 0 and 1; the two shares in a
    race add up to 1 and each race and draw has exactly one winner."""
    draw_check = pa.Check.in_range(1, n_draws) if n_draws else pa.Check.ge(0)
    keys = ["draw", "state", "model"]
    return pa.DataFrameSchema({
        "draw": pa.Column(int, draw_check),
        "state": pa.Column(str, pa.Check.isin(states)),
        "model": pa.Column(str, pa.Check.isin(models)),
        "party": pa.Column(str, pa.Check.isin(PARTIES)),
        "share": pa.Column(float, prob, nullable=False),
        "winner": pa.Column(str, pa.Check.isin(PARTIES)),
    }, unique=keys + ["party"], checks=[
        pa.Check(lambda d: (d.groupby(keys)["share"].sum() - 1).abs().lt(1e-9).all(),
                 error="the two shares in a race should add up to 1"),
        pa.Check(lambda d: d.groupby(keys).size().eq(2).all(), error="each race should have exactly two shares"),
        pa.Check(lambda d: d.groupby(keys)["winner"].nunique().eq(1).all(),
                 error="each race and draw should have one winner"),
    ])


def summary_of(err):
    """Short, readable list of what failed in a pandera SchemaErrors exception."""
    cases = err.failure_cases.assign(check=lambda d: d["check"].astype(str).str.slice(0, 70),
                                     column=lambda d: d["column"].fillna("(table)"))
    out = cases.groupby(["column", "check"]).agg(failures=("failure_case", "size"),
                                                 example=("failure_case", "first"))
    return out.to_string()


def validate(schema, df):
    """Validate with lazy=True (collect every failure, not only the first) and raise one
    AssertionError with the summary, so pytest reports it like any other failed test."""
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        raise AssertionError("Schema check failed:\n" + summary_of(err)) from None
