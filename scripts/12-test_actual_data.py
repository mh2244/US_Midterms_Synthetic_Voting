#### Preamble ####
# Purpose: Tests the REAL (non-simulated) candidates, personas, prompts, model answers, scores
#          and forecast. It follows the style of scripts/01-test_simulated_data.py.
# Author: (group)
# Date: 22 October 2026
# License: MIT
# Pre-requisites: Run scripts 06 to 11 first (script 09 can be run with --mock).
# Any other information needed? Run from the project root:  python scripts/12-test_actual_data.py
#   Each test is an `assert`. If one fails, the script stops and prints the message after the
#   comma. Fix the data (or the earlier script), not the test.
#
# WHY separate tests for actual data: the assignment asks for test suites for both simulated
# and actual data, covering missing answers and shares that do not add up to 100%.


#### Workspace setup ####
import numpy as np
import pandas as pd

import config

nominees = pd.read_csv(config.NOMINEES_FILE)
personas = config.read_table(config.PERSONAS_FILE)
prompts = config.read_table(config.PROMPTS_FILE)
responses = config.read_table(config.RESPONSES_FILE)
scores = config.read_table(config.SCORES_FILE)
forecast = config.read_table(config.FORECAST_FILE)
print(f"Testing run {config.RUN}")

#### Settings used by the tests ####
n_races = 35
dem_not_up = 34          # seats not up in 2026 (same numbers as the simulation; check them)
rep_not_up = 31
max_problem_rate = 0.10  # largest share of empty or failed answers we accept per model

allowed_labels = {
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
models = sorted(responses["model"].unique())
n_per_state = personas.groupby("state").size().iloc[0]

#### Test candidates ####
# 1. 35 races, one row each, with two different names
assert len(nominees) == n_races and nominees["state"].is_unique, "Expected 35 races, one row per state"
assert nominees[["left_name", "rep_name"]].notna().all().all(), "A race has no left or Republican name"
assert (nominees["left_name"] != nominees["rep_name"]).all(), "Left and Republican name are the same"

# 2. Labels
assert nominees["race_type"].isin(["regular", "special"]).all(), "Unexpected race_type"
assert (nominees["race_type"] == "special").sum() == 2, "Expected 2 special elections (FL, OH)"
assert nominees["left_party"].isin(["Democratic", "Independent"]).all(), "Unexpected left_party"

print("Candidates: all tests passed")

#### Test personas ####
# 3. Same number of personas in every state, one id each
assert set(personas["state"]) == set(nominees["state"]), "Personas and races cover different states"
assert (personas.groupby("state").size() == n_per_state).all(), "States have different numbers of personas"
assert personas["persona_id"].is_unique, "Duplicate persona_id"

# 4. A repeated CES respondent stays in one state and has identical answers each time
assert (personas.groupby("caseid")["state"].nunique() == 1).all(), "A caseid appears in two states"
for column in allowed_labels:
    assert (personas.groupby("caseid")[column].nunique() == 1).all(), "A repeated caseid differs in " + column
print("Personas that repeat an earlier draw:", int(personas.duplicated("caseid").sum()), "of", len(personas))

# 5. No missing values; only allowed labels
assert personas[list(allowed_labels)].notna().all().all(), "Missing value in a persona variable"
for column in allowed_labels:
    assert personas[column].isin(allowed_labels[column]).all(), "Unexpected label in " + column

# 6. The 2024 presidential vote is not in the persona text
assert not personas["persona_text"].str.contains("Trump|Harris|president", case=False).any(), \
    "Presidential vote leaked into persona_text"
print("Personas: all tests passed")

#### Test prompts ####
# 7. One row per persona and design; each prompt names both candidates of its race
assert prompts["prompt_id"].is_unique, "Duplicate prompt_id"
assert set(prompts["persona_id"]) == set(personas["persona_id"]), "Prompts and personas differ"
merged = prompts.merge(nominees, on="state")
names_ok = merged.apply(lambda r: r["left_name"] in r["prompt_text"] and r["rep_name"] in r["prompt_text"], axis=1)
assert names_ok.all(), "A prompt does not name both candidates of its race"
print("Prompts: all tests passed")

#### Test responses ####
# 8. Complete grid: every prompt answered once per model and repeat
n_repeats = responses["repeat"].nunique()
assert len(responses) == len(prompts) * len(models) * n_repeats, "Wrong number of responses"
# Every prompt and answer belongs to this run (runs must never mix)
assert prompts["prompt_id"].str.endswith("|" + config.RUN).all(), "A prompt from another run"
assert responses["prompt_id"].str.endswith("|" + config.RUN).all(), "An answer from another run"
assert set(responses["prompt_id"]) <= set(prompts["prompt_id"]), "An answer has no matching prompt"
# Dated runs must contain news; the baseline must not
assert prompts["news"].eq(config.NEWS_RUN).all(), "News setting does not match the run"
assert not responses.duplicated(["prompt_id", "model", "repeat"]).any(), "A prompt was answered twice"

# 9. Share of empty or failed answers per model is below the limit
responses["error"] = responses["error"].fillna("")
problem = ((responses["response_text"].fillna("").str.strip() == "") | (responses["error"] != ""))
problem_rate = problem.groupby(responses["model"]).mean()
print("Empty or failed answer rate by model:")
print(problem_rate.round(3))
assert (problem_rate < max_problem_rate).all(), "A model has too many empty or failed answers"

# 10. The exact model id and query time were saved
assert responses["model_returned"].fillna("").ne("").all() or responses["error"].ne("").any(), "model_returned missing"
assert responses["queried_at_utc"].notna().all(), "queried_at_utc missing"
print("Responses: all tests passed")

#### Test scores ####
# 11. Probabilities are between 0 and 1 and add up to 1 where they exist
ok = scores[scores["p_left"].notna()]
assert ok[["p_left", "p_rep", "p_other"]].ge(0).all().all(), "Negative probability"
assert np.isclose(ok[["p_left", "p_rep", "p_other"]].sum(axis=1), 1).all(), "Probabilities do not add up to 1"

# 12. Only rows with status "ok" have probabilities, and every row has a status
assert scores["status"].isin(["ok", "refusal", "empty", "error", "bad_shares"]).all(), "Unexpected status"
assert (scores.loc[scores["status"] != "ok", "p_left"].isna()).all(), "A row that is not ok has a probability"
assert (scores.loc[scores["status"] == "ok", "p_left"].notna()).all(), "An ok row has no probability"

# 13. Every state has at least one scored persona for every model (design B)
counts = scores[(scores["design"] == "B") & scores["p_left"].notna()].groupby(["state", "model"]).size()
assert len(counts) == n_races * len(models), "A state and model pair has no scored answers"
print("Scores: all tests passed")

#### Test forecast ####
keys = ["draw", "state", "model"]

# 14. No missing shares; all between 0 and 1; shares add up to 1
assert forecast["share"].notna().all(), "Missing share in forecast"
assert forecast["share"].between(0, 1).all(), "Share outside 0 to 1"
assert np.isclose(forecast.groupby(keys)["share"].sum(), 1).all(), "Shares do not add up to 1"

# 15. One winner per race, draw and model, and it is the party with the larger share
assert (forecast.groupby(keys)["winner"].nunique() == 1).all(), "More than one winner"
#     (exact 50/50 ties are decided by a coin flip in script 11, so they are skipped here)
untied = forecast.groupby(keys)["share"].transform("nunique") > 1
top_rows = forecast[untied]
top = top_rows.loc[top_rows.groupby(keys)["share"].idxmax()]
assert (top["party"] == top["winner"]).all(), "Winner is not the party with the largest share"
print("Exact ties handled by coin flip:", int((~untied).sum() // 2))

# 16. The winner is a party running in that race (the left party of the state, or Republican)
allowed = nominees.set_index("state")["left_party"]
winners = forecast[["state", "winner"]].drop_duplicates()
assert winners.apply(lambda r: r["winner"] in (allowed[r["state"]], "Republican"), axis=1).all(), \
    "Winner is not running in that state"

# 17. Every state and model has the same number of draws
draws = forecast.groupby(["state", "model"])["draw"].nunique()
assert len(draws) == n_races * len(models), "A state and model pair is missing"
assert draws.nunique() == 1, "States have different numbers of draws"

# 18. Every draw fills 100 seats
races_per_draw = forecast[["model", "draw", "state"]].drop_duplicates().groupby(["model", "draw"]).size()
assert (races_per_draw + dem_not_up + rep_not_up == 100).all(), "Seats do not add up to 100"
print("Forecast: all tests passed")
print("\nAll tests passed.")
