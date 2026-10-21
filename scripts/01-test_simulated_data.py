#### Preamble ####
# Purpose: Tests the simulated candidates, personas, LLM responses and forecast.
# Author: Kexin Liu
# Date: 21 October 2026
# Contact: ws1nn2lj3@gmail.com
# License: MIT
# Pre-requisites: Run scripts/00-simulate_data.py first.
# Any other information needed? Run this from the project's root folder.
#   Each test is an `assert`. If a test fails, the script stops and prints
#   the message after the comma. 


#### Workspace setup ####
import numpy as np
import pandas as pd

candidates = pd.read_csv("data/00-simulated_data/simulated_candidates.csv")
personas = pd.read_csv("data/00-simulated_data/simulated_personas.csv")
responses = pd.read_csv("data/00-simulated_data/simulated_responses.csv")
forecast = pd.read_parquet("data/00-simulated_data/simulated_forecast.parquet")


#### Settings used by the tests ####
n_races = 35
n_per_state = 100
n_draws = 1000
models = ["gpt_a", "gpt_b", "claude", "deepseek"]

# Seats not up in 2026 (check these before using real data)
dem_not_up = 34
rep_not_up = 31

# Largest share of unusable answers (NA or junk) we accept from one model
max_problem_rate = 0.10

# Fewest answers naming a candidate we accept per state and model
min_valid_answers = 50

all_parties = ["Democratic", "Republican", "Independent", "Libertarian"]

# Allowed labels for the 7 persona variables (same as in the simulation)
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
                 "Very conservative", "Not sure"],}

# Which parties run in which state. Several tests use this table.
state_parties = candidates[["state", "party"]].drop_duplicates()


#### Test candidates ####
# 1. There are exactly 35 races, one per state
assert candidates["state"].nunique() == n_races, "Wrong number of races"

# 2. race_type only has the two expected values
assert candidates["race_type"].isin(["regular", "special"]).all(), "Unexpected race_type"

# 3. Each state has only one race_type (no state has two races this cycle)
assert (candidates.groupby("state")["race_type"].nunique() == 1).all(), "A state has two race types"

# 4. Parties only use the labels the forecast looks for
assert candidates["party"].isin(all_parties).all(), "Unexpected party label"

# 5. Every race has at least two candidates
assert (candidates.groupby("state").size() >= 2).all(), "A race has fewer than 2 candidates"

# 6. Every race has a Democratic or Republican candidate
main_parties = candidates[candidates["party"].isin(["Democratic", "Republican"])]
assert main_parties["state"].nunique() == n_races, "A race has no major-party candidate"

# 7. No candidate is listed twice in the same state
assert not candidates.duplicated(["state", "name"]).any(), "Duplicate candidate name in a state"

print("Candidates: all tests passed")


#### Test personas ####
# 8. Every state has exactly 100 personas
assert personas["state"].nunique() == n_races, "Wrong number of states in personas"
assert (personas.groupby("state").size() == n_per_state).all(), "A state does not have 100 personas"

# 9. Each persona has its own persona_id.
#    caseid CAN repeat, because we sample with replacement within each state.
assert personas["persona_id"].is_unique, "Duplicate persona_id in personas"

# 9b. A repeated CES respondent stays in one state
assert (personas.groupby("caseid")["state"].nunique() == 1).all(), "A caseid appears in two states"

# 9c. A repeated CES respondent has identical answers every time they appear
for column in allowed_labels:
    same = personas.groupby("caseid")[column].nunique() == 1
    assert same.all(), "A repeated caseid has different values for " + column

# Show how often people were drawn more than once (useful for the write-up)
n_repeats = personas.duplicated("caseid").sum()
print("Personas that repeat an earlier draw:", n_repeats, "of", len(personas))

# 10. None of the 7 persona variables is missing
assert personas[list(allowed_labels)].notna().all().all(), "Missing value in a persona variable"

# 11. Each variable only uses its allowed labels
for column in allowed_labels:
    assert personas[column].isin(allowed_labels[column]).all(), "Unexpected label in " + column

# 12. Every persona lives in a state that has a race
assert personas["state"].isin(candidates["state"]).all(), "A persona's state has no race"

print("Personas: all tests passed")


#### Test responses ####
# 13. Exactly the 4 models we planned
assert sorted(responses["model"].unique()) == sorted(models), "Wrong set of models"

# 14. Complete grid: every persona answered once per model
assert len(responses) == len(personas) * len(models), "Wrong number of responses"
assert not responses.duplicated(["persona_id", "model"]).any(), "A persona answered twice for one model"
assert set(responses["persona_id"]) == set(personas["persona_id"]), "Responses and personas have different people"

# Mark each answer that names a party running in that persona's state
check = responses.merge(state_parties, left_on=["state", "vote_choice"],
                        right_on=["state", "party"], how="left")
check["names_candidate"] = check["party"].notna()

# 15. Nobody voted for a party with no candidate in their state
#     (for example, "Democratic" in a state with no Democrat)
party_answers = check[check["vote_choice"].isin(all_parties)]
assert party_answers["names_candidate"].all(), "An answer names a party not running in that state"

# 16. Each model's share of unusable answers is below our limit.
#     Usable means naming a candidate or saying "Undecided". NA and junk are unusable.
check["usable"] = check["names_candidate"] | (check["vote_choice"] == "Undecided")
problem_rate = 1 - check.groupby("model")["usable"].mean()
print("Unusable answer rate by model:")
print(problem_rate.round(3))
assert (problem_rate < max_problem_rate).all(), "A model has too many unusable answers"

# 17. Every state has enough answers naming a candidate, for every model
valid_counts = check.groupby(["state", "model"])["names_candidate"].sum()
assert len(valid_counts) == n_races * len(models), "A state and model pair has no responses"
assert (valid_counts >= min_valid_answers).all(), "Too few valid answers in a state for a model"

print("Responses: all tests passed")


#### Test forecast ####
keys = ["draw", "state", "model"]

# 18. No missing shares, so NAs did not flow through
assert forecast["share"].notna().all(), "Missing share in forecast"

# 19. Every share is between 0 and 1
assert forecast["share"].between(0, 1).all(), "Share outside 0 to 1"

# 20. Shares add up to 1 in every race, draw and model.
#     np.isclose allows for tiny rounding errors like 0.9999999999999999.
share_sums = forecast.groupby(keys)["share"].sum()
assert np.isclose(share_sums, 1).all(), "Shares do not add up to 1"

# 21. Each race, draw and model has exactly one winner
assert (forecast.groupby(keys)["winner"].nunique() == 1).all(), "More than one winner"

# 22. The winner is the party with the largest share.
#     idxmax finds the row with the largest share in each group.
top = forecast.loc[forecast.groupby(keys)["share"].idxmax()]
assert (top["party"] == top["winner"]).all(), "Winner is not the party with the largest share"

# 23. The winner is a party running in that state
winners = forecast[["state", "winner"]].drop_duplicates()
winners = winners.merge(state_parties, left_on=["state", "winner"],
                        right_on=["state", "party"], how="left")
assert winners["party"].notna().all(), "Winner is not running in that state"

# 24. Every state and model has all 1,000 draws
draws = forecast.groupby(["state", "model"])["draw"].nunique()
assert len(draws) == n_races * len(models), "A state and model pair is missing"
assert (draws == n_draws).all(), "A state and model pair does not have 1,000 draws"

# 25. Every draw gives a full Senate of 100 seats
races_per_draw = forecast[["model", "draw", "state"]].drop_duplicates().groupby(["model", "draw"]).size()
assert (races_per_draw + dem_not_up + rep_not_up == 100).all(), "Seats do not add up to 100"

print("Forecast: all tests passed")
print("\nAll tests passed.")
