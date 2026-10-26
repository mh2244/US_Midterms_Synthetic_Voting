#### Preamble ####
# Purpose: Tests the simulated races, personas, scored answers and forecast.
# Author: Kexin Liu
# Date: 25 October 2026
# Contact: ws1nn2lj3@gmail.com
# License: MIT
# Pre-requisites: Run scripts/00-simulate_data.py first.
# Any other information needed? Run from the project root:
#   python scripts/01-test_simulated_data.py
#   Each test is an `assert`. If a test fails, the script stops and prints the message
#   after the comma. The same checks are run on the real data in 12-test_actual_data.py.


#### Workspace setup ####
import numpy as np
import pandas as pd

nominees = pd.read_csv("data/00-simulated_data/simulated_nominees.csv")
candidates = pd.read_csv("data/00-simulated_data/simulated_candidates.csv")
personas = pd.read_csv("data/00-simulated_data/simulated_personas.csv")
scores = pd.read_csv("data/00-simulated_data/simulated_scores.csv")
forecast = pd.read_parquet("data/00-simulated_data/simulated_forecast.parquet")


#### Settings used by the tests ####
n_races = 35
n_per_state = 100
n_draws = 1000
models = ["haiku", "luna", "deepseek", "gemini"]   # same as MODELS in config.py
n_repeats = 3

# Seats not up in 2026 (same as script 11; check before the final run)
dem_not_up = 34
rep_not_up = 31

# Largest share of answers that cannot be scored that we accept from one model
max_problem_rate = 0.10

# Fewest scored personas we accept in a state for one model
min_scored_personas = 50

allowed_status = ["ok", "refusal", "empty", "error", "bad_shares"]

# Allowed labels for the 7 persona variables (same as the simulation and script 07)
allowed_labels = {
    "age_group": ["18-29", "30-44", "45-64", "65+"],
    "family_income": [
        "Less than $10,000", "$10,000 - $19,999", "$20,000 - $29,999",
        "$30,000 - $39,999", "$40,000 - $49,999", "$50,000 - $59,999",
        "$60,000 - $69,999", "$70,000 - $79,999", "$80,000 - $99,999",
        "$100,000 - $119,999", "$120,000 - $149,999", "$150,000 - $199,999",
        "$200,000 - $249,999", "$250,000 - $349,999", "$350,000 - $499,999",
        "$500,000 or more", "Prefer not to say",
    ],
    "education": ["No HS", "HS grad", "Some college", "2-year", "4-year", "Post-grad"],
    "gender": ["Man", "Woman", "Non-binary", "Other"],
    "race_5": ["White", "Black", "Hispanic", "Asian", "Other"],
    "party_id_7": ["Strong Dem", "Weak Dem", "Lean Dem", "Independent",
                   "Lean Rep", "Weak Rep", "Strong Rep", "Not sure"],
    "ideology": ["Very liberal", "Liberal", "Moderate", "Conservative",
                 "Very conservative", "Not sure"],
}

# The left party of each state (Democratic, or Independent where there is no Democrat)
left_party_of = dict(zip(nominees["state"], nominees["left_party"]))


#### Test races ####
# 1. Exactly 35 races, one row per state
assert len(nominees) == n_races, "Wrong number of races"
assert nominees["state"].is_unique, "A state has more than one race"

# 2. race_type only has the two expected values, and there are 2 special elections
assert nominees["race_type"].isin(["regular", "special"]).all(), "Unexpected race_type"
assert (nominees["race_type"] == "special").sum() == 2, "Expected 2 special elections (FL, OH)"

# 3. The left candidate is a Democrat or an Independent
assert nominees["left_party"].isin(["Democratic", "Independent"]).all(), "Unexpected left_party"

# 4. Every race has two different names
assert nominees[["left_name", "rep_name"]].notna().all().all(), "A race is missing a name"
assert (nominees["left_name"] != nominees["rep_name"]).all(), "Left and Republican names are the same"

# 5. The long candidate file has exactly two rows per race: the left party and Republican
assert (candidates.groupby("state").size() == 2).all(), "A race does not have exactly 2 candidates"
assert (candidates[candidates["party"] == "Republican"].groupby("state").size() == 1).all(), \
    "A race does not have exactly one Republican"
left_rows = candidates[candidates["party"] != "Republican"]
assert (left_rows["party"] == left_rows["state"].map(left_party_of)).all(), \
    "Long and wide candidate files disagree on the left party"

print("Races: all tests passed")


#### Test personas ####
# 6. Every state has exactly 100 personas
assert set(personas["state"]) == set(nominees["state"]), "Personas and races cover different states"
assert (personas.groupby("state").size() == n_per_state).all(), "A state does not have 100 personas"

# 7. Each persona has its own persona_id.
#    caseid CAN repeat, because we sample with replacement within each state.
assert personas["persona_id"].is_unique, "Duplicate persona_id in personas"

# 8. A repeated CES respondent stays in one state
assert (personas.groupby("caseid")["state"].nunique() == 1).all(), "A caseid appears in two states"

# 9. A repeated CES respondent has identical values every time they appear
for column in allowed_labels:
    same = personas.groupby("caseid")[column].nunique() == 1
    assert same.all(), "A repeated caseid has different values for " + column

# 10. None of the 7 persona variables is missing
assert personas[list(allowed_labels)].notna().all().all(), "Missing value in a persona variable"

# 11. Each variable only uses its allowed labels
for column in allowed_labels:
    assert personas[column].isin(allowed_labels[column]).all(), "Unexpected label in " + column

# 12. The weight was dropped after sampling, so nobody weights twice
assert "vvweight_post" not in personas.columns, "Weight column still in personas"

# Show how often people were drawn more than once (useful for the write-up)
n_repeats_drawn = personas.duplicated("caseid").sum()
print("Personas that repeat an earlier draw:", n_repeats_drawn, "of", len(personas))

print("Personas: all tests passed")


#### Test scored answers ####
# 13. Exactly the planned models and repeats
assert sorted(scores["model"].unique()) == sorted(models), "Wrong set of models"
assert sorted(scores["repeat"].unique()) == list(range(1, n_repeats + 1)), "Wrong set of repeats"

# 14. Complete grid: every persona answered once per model and repeat
assert len(scores) == len(personas) * len(models) * n_repeats, "Wrong number of answers"
assert not scores.duplicated(["persona_id", "model", "repeat"]).any(), \
    "A persona answered twice for one model and repeat"
assert set(scores["persona_id"]) == set(personas["persona_id"]), "Answers and personas have different people"

# 15. Every answer has a known status
assert scores["status"].isin(allowed_status).all(), "Unexpected status"

# 16. Only "ok" answers have probabilities, and every "ok" answer has them.
#     This is how missing answers are kept out of the forecast.
is_ok = scores["status"] == "ok"
p_cols = ["p_left", "p_rep", "p_other"]
assert scores.loc[is_ok, p_cols].notna().all().all(), "An ok answer is missing a probability"
assert scores.loc[~is_ok, p_cols].isna().all().all(), "An answer that is not ok has a probability"

# 17. Probabilities are between 0 and 1 and add up to 1 (np.isclose allows tiny rounding errors)
ok = scores[is_ok]
assert ok[p_cols].ge(0).all().all() and ok[p_cols].le(1).all().all(), "Probability outside 0 to 1"
assert np.isclose(ok[p_cols].sum(axis=1), 1).all(), "Probabilities do not add up to 1"

# 18. The readable label matches the largest probability and the parties in that race
expected = []
for _, row in ok.iterrows():
    labels = [left_party_of[row["state"]], "Republican", "Undecided"]
    expected.append(labels[int(np.argmax(row[p_cols].to_numpy(dtype=float)))])
assert (ok["vote_choice"].to_numpy() == np.array(expected)).all(), "vote_choice does not match the probabilities"
assert scores.loc[~is_ok, "vote_choice"].isna().all(), "An answer that is not ok has a vote_choice"

# 19. Each model's share of answers that cannot be scored is below our limit
problem_rate = 1 - is_ok.groupby(scores["model"]).mean()
print("Unscorable answer rate by model:")
print(problem_rate.round(3))
assert (problem_rate < max_problem_rate).all(), "A model has too many unscorable answers"

# 20. Every state has enough scored personas for every model
scored_personas = ok.groupby(["state", "model"])["persona_id"].nunique()
assert len(scored_personas) == n_races * len(models), "A state and model pair has no scored answers"
assert (scored_personas >= min_scored_personas).all(), "Too few scored personas in a state for a model"

print("Scored answers: all tests passed")


#### Test forecast ####
keys = ["draw", "state", "model"]

# 21. No missing shares, so NAs did not flow through
assert forecast["share"].notna().all(), "Missing share in forecast"

# 22. Every share is between 0 and 1
assert forecast["share"].between(0, 1).all(), "Share outside 0 to 1"

# 23. Shares add up to 1 in every race, draw and model
share_sums = forecast.groupby(keys)["share"].sum()
assert np.isclose(share_sums, 1).all(), "Shares do not add up to 1"

# 24. Each race, draw and model has exactly two rows: the left party and Republican
assert (forecast.groupby(keys).size() == 2).all(), "A race does not have exactly two shares"
expected_party = np.where(forecast["party"] == "Republican", "Republican", forecast["state"].map(left_party_of))
assert (forecast["party"] == expected_party).all(), "A share is for a party not running in that state"

# 25. Each race, draw and model has exactly one winner
assert (forecast.groupby(keys)["winner"].nunique() == 1).all(), "More than one winner"

# 26. The winner is the party with the larger share.
#     Exact 50/50 ties are skipped, because script 11 breaks them with a coin flip.
untied = forecast.groupby(keys)["share"].transform("nunique") > 1
top_rows = forecast[untied]
top = top_rows.loc[top_rows.groupby(keys)["share"].idxmax()]
assert (top["party"] == top["winner"]).all(), "Winner is not the party with the larger share"

# 27. Every state and model has all 1,000 draws
draws = forecast.groupby(["state", "model"])["draw"].nunique()
assert len(draws) == n_races * len(models), "A state and model pair is missing"
assert (draws == n_draws).all(), "A state and model pair does not have 1,000 draws"

# 28. Every draw gives a full Senate of 100 seats
races_per_draw = forecast[["model", "draw", "state"]].drop_duplicates().groupby(["model", "draw"]).size()
assert (races_per_draw + dem_not_up + rep_not_up == 100).all(), "Seats do not add up to 100"

print("Forecast: all tests passed")
print("\nAll tests passed.")
