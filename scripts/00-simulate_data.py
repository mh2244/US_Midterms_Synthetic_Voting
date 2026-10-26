#### Preamble ####
# Purpose: Simulates the races, the persona sample, the scored model answers and the
#          forecast for our 2026 US Senate silicon-sampling project.
# Author: Kexin Liu
# Date: 25 October 2026
# Contact: ws1nn2lj3@gmail.com
# License: MIT
# Pre-requisites: pip install pandas numpy pyarrow
# Any other information needed? Run from the project root:
#   python scripts/00-simulate_data.py
#   The simulated files have the same columns as the real files made by scripts 06, 07,
#   10 and 11, so the same tests can check both. Numbers marked "same as config.py"
#   must be changed in both places if the plan changes.


#### Workspace setup ####
import os
import numpy as np
import pandas as pd

np.random.seed(2026)          # same as SEED in config.py

os.makedirs("data/00-simulated_data", exist_ok=True)


#### Settings ####
n_per_state = 100             # same as N_PER_STATE in config.py
models = ["haiku", "luna", "deepseek", "gemini"]   # short model names, same as MODELS in config.py
n_repeats = 3                 # same as N_REPEATS in config.py
n_draws = 1000                # same as N_DRAWS in config.py
national_sd = 0.03            # same as NATIONAL_SD in config.py


#### Races ####
# Script 06 keeps two candidates per race: the main "left" candidate (the Democrat, or an
# Independent where there is no Democrat) and the Republican. Other candidates are only
# counted in n_others. We copy that here, including the three races where the left
# candidate is an Independent (ID, NE, SD) and the races with extra candidates.
regular_states = [
    "AL", "AK", "AR", "CO", "DE", "GA", "ID", "IL", "IA", "KS", "KY",
    "LA", "ME", "MA", "MI", "MN", "MS", "MT", "NE", "NH", "NJ", "NM",
    "NC", "OK", "OR", "RI", "SC", "SD", "TN", "TX", "VA", "WV", "WY",
]
special_states = ["FL", "OH"]
all_states = regular_states + special_states

independent_left = ["ID", "NE", "SD"]

rows = []
for state in all_states:
    left_party = "Independent" if state in independent_left else "Democratic"
    rows.append({
        "state": state,
        "race_type": "special" if state in special_states else "regular",
        "left_name": state + " " + left_party + " candidate",   # fake names
        "left_party": left_party,
        "rep_name": state + " Republican candidate",
        "n_others": np.random.randint(0, 4),
    })

nominees = pd.DataFrame(rows)

# Long version: one row per candidate, like data/02-analysis_data/candidates_clean.csv
rows = []
for _, race in nominees.iterrows():
    rows.append({"state": race["state"], "race_type": race["race_type"],
                 "name": race["left_name"], "party": race["left_party"]})
    rows.append({"state": race["state"], "race_type": race["race_type"],
                 "name": race["rep_name"], "party": "Republican"})

candidates = pd.DataFrame(rows)


#### Categories ####
# Labels copied from the CES columns we use (same as script 07)
age_groups = ["18-29", "30-44", "45-64", "65+"]
family_incomes = [
    "Less than $10,000", "$10,000 - $19,999", "$20,000 - $29,999",
    "$30,000 - $39,999", "$40,000 - $49,999", "$50,000 - $59,999",
    "$60,000 - $69,999", "$70,000 - $79,999", "$80,000 - $99,999",
    "$100,000 - $119,999", "$120,000 - $149,999", "$150,000 - $199,999",
    "$200,000 - $249,999", "$250,000 - $349,999", "$350,000 - $499,999",
    "$500,000 or more", "Prefer not to say",
]
educations = ["No HS", "HS grad", "Some college", "2-year", "4-year", "Post-grad"]
genders = ["Man", "Woman", "Non-binary", "Other"]
race_groups = ["White", "Black", "Hispanic", "Asian", "Other"]
ideologies = ["Very liberal", "Liberal", "Moderate", "Conservative",
              "Very conservative", "Not sure"]

# party_id_7 grouped into three "sides". Leaners go with the party they lean to.
party_labels = {
    "dem":    ["Strong Dem", "Weak Dem", "Lean Dem"],
    "rep":    ["Strong Rep", "Weak Rep", "Lean Rep"],
    "middle": ["Independent", "Not sure"],
}

# Ideology depends on side, so our fake people are plausible
ideology_probs = {
    "dem":    [0.25, 0.40, 0.25, 0.04, 0.01, 0.05],
    "rep":    [0.01, 0.04, 0.25, 0.40, 0.25, 0.05],
    "middle": [0.05, 0.10, 0.45, 0.10, 0.05, 0.25],
}


#### Simulate a CES pool ####
# A fake version of the cleaned CES: 2024 validated voters in each state (script 07 keeps
# only these), each with a vvweight_post (the weight script 07 uses). Pool sizes differ by state; the smallest
# are near Wyoming's ~54, so drawing 100 with replacement repeats people.
rows = []
caseid = 1

for state in all_states:
    # Give each state its own partisan lean
    p_dem = np.random.uniform(0.30, 0.50)
    p_mid = 0.20
    p_rep = 1 - p_dem - p_mid

    pool_size = np.random.randint(50, 1000)

    for i in range(pool_size):
        side = np.random.choice(["dem", "rep", "middle"], p=[p_dem, p_rep, p_mid])

        rows.append({
            "caseid": caseid,
            "state": state,
            "vvweight_post": np.random.uniform(0.1, 3),
            "age_group": np.random.choice(age_groups, p=[0.15, 0.25, 0.35, 0.25]),
            "family_income": np.random.choice(family_incomes),
            "education": np.random.choice(educations, p=[0.05, 0.25, 0.20, 0.10, 0.25, 0.15]),
            "gender": np.random.choice(genders, p=[0.48, 0.50, 0.01, 0.01]),
            "race_5": np.random.choice(race_groups, p=[0.65, 0.12, 0.12, 0.05, 0.06]),
            "party_id_7": np.random.choice(party_labels[side]),
            "ideology": np.random.choice(ideologies, p=ideology_probs[side]),
        })
        caseid = caseid + 1

ces_pool = pd.DataFrame(rows)


#### Sample personas ####
# In each state, draw 100 people WITH replacement, with probability proportional to
# vvweight_post (as script 07 does). The same CES respondent can be drawn more than once,
# so caseid can repeat; persona_id gives every draw its own id. Because people were drawn
# in proportion to the weight, each persona already counts equally, so we drop the weight
# to avoid weighting twice.
samples = []

for state in all_states:
    state_pool = ces_pool[ces_pool["state"] == state]
    probs = state_pool["vvweight_post"] / state_pool["vvweight_post"].sum()

    picked = np.random.choice(state_pool.index, size=n_per_state, replace=True, p=probs)
    sample = state_pool.loc[picked].copy()

    sample["persona_id"] = [state + "_" + str(i + 1) for i in range(n_per_state)]
    samples.append(sample)

personas = pd.concat(samples).drop(columns="vvweight_post")

# Put persona_id first so the table is easy to read
personas = personas[["persona_id"] + [c for c in personas.columns if c != "persona_id"]]


#### Simulate scored answers ####
# In the real pipeline, script 09 gets a free-text answer and script 10 turns it into three
# probabilities: p_left (votes for the left candidate), p_rep (votes Republican) and p_other
# (someone else, undecided, or not voting). They add up to 1. Answers that are refused,
# empty or failed get a status other than "ok" and NO probabilities.
# Here we skip the text and simulate the probabilities directly.

# Average probabilities (p_left, p_rep, p_other) for each side
base_probs = {
    "dem":    [0.80, 0.08, 0.12],
    "rep":    [0.08, 0.80, 0.12],
    "middle": [0.35, 0.35, 0.30],
}

# How often each model's answer cannot be scored. Planted on purpose for the tests.
# The rates differ by model so the tests can show that they are reported per model.
status_rates = {
    "haiku":    {"refusal": 0.03, "empty": 0.01, "error": 0.01},
    "luna":     {"refusal": 0.02, "empty": 0.01, "error": 0.01},
    "deepseek": {"refusal": 0.01, "empty": 0.02, "error": 0.02},
    "gemini":   {"refusal": 0.01, "empty": 0.04, "error": 0.01},
}

# Look up which side each party_id_7 label belongs to
side_of = {}
for side in party_labels:
    for label in party_labels[side]:
        side_of[label] = side

left_party_of = dict(zip(nominees["state"], nominees["left_party"]))

rows = []

for model in models:
    for repeat in range(1, n_repeats + 1):
        for _, person in personas.iterrows():
            side = side_of[person["party_id_7"]]

            # Random probabilities that add up to 1 and centre on the side's base_probs.
            # A larger number (20) means answers stay closer to the base.
            p_left, p_rep, p_other = np.random.dirichlet(20 * np.array(base_probs[side]))

            # Plant answers that cannot be scored
            rates = status_rates[model]
            u = np.random.uniform()
            if u < rates["refusal"]:
                status = "refusal"
            elif u < rates["refusal"] + rates["empty"]:
                status = "empty"
            elif u < rates["refusal"] + rates["empty"] + rates["error"]:
                status = "error"
            else:
                status = "ok"

            if status != "ok":
                p_left, p_rep, p_other = np.nan, np.nan, np.nan
                vote_choice = None
            else:
                # Readable label = the largest probability (as script 10 does)
                labels = [left_party_of[person["state"]], "Republican", "Undecided"]
                vote_choice = labels[int(np.argmax([p_left, p_rep, p_other]))]

            rows.append({
                "persona_id": person["persona_id"],
                "caseid": person["caseid"],
                "state": person["state"],
                "design": "B",
                "model": model,
                "repeat": repeat,
                "status": status,
                "p_left": p_left,
                "p_rep": p_rep,
                "p_other": p_other,
                "vote_choice": vote_choice,
            })

scores = pd.DataFrame(rows)


#### Simulate forecast ####
# Same steps as script 11:
#   1. Average each persona's probabilities over repeats (scorable answers only).
#   2. Resample a state's personas with replacement 1,000 times.
#   3. Left share = mean p_left / (mean p_left + mean p_rep); the Republican gets the rest.
#   4. Add a national error: one random shift per draw, shared by ALL states, so states
#      move together (a polling-style miss). Then keep the share between 0 and 1.
# So each race has exactly two shares, and they add up to 1.
ok = scores[scores["status"] == "ok"]
per_persona = ok.groupby(["state", "persona_id", "model"])[["p_left", "p_rep"]].mean().reset_index()

rows = []

for model in models:
    # One shift per draw for this model. Draw 1 uses shift[0] in every state, and so on.
    shift = np.random.normal(0, national_sd, n_draws)

    for state in all_states:
        sub = per_persona[(per_persona["model"] == model) & (per_persona["state"] == state)]
        left = sub["p_left"].to_numpy()
        rep = sub["p_rep"].to_numpy()
        left_party = left_party_of[state]

        for draw in range(1, n_draws + 1):
            idx = np.random.randint(0, len(sub), size=len(sub))   # resample personas
            mean_left = left[idx].mean()
            mean_rep = rep[idx].mean()
            share = mean_left / (mean_left + mean_rep)
            share = min(max(share + shift[draw - 1], 0), 1)

            # Script 11 breaks an exact 50/50 tie with a coin flip. With 100 personas and
            # continuous probabilities an exact tie practically never happens.
            winner = left_party if share > 0.5 else "Republican"

            rows.append({"draw": draw, "state": state, "model": model,
                         "party": left_party, "share": share, "winner": winner})
            rows.append({"draw": draw, "state": state, "model": model,
                         "party": "Republican", "share": 1 - share, "winner": winner})

forecast = pd.DataFrame(rows)


#### Quick summary ####
# Seats not up in 2026: 34 held by Democrats or independents who caucus with them,
# 31 by Republicans (same as script 11). Check before the final run.
dem_not_up = 34
rep_not_up = 31

# One row per model, draw and state, then count the seats each side wins (as script 11).
# Independent wins (possible in ID, NE, SD) are counted separately, not as Democratic.
results = forecast[["model", "draw", "state", "winner"]].drop_duplicates()
results["dem"] = results["winner"] == "Democratic"
results["ind"] = results["winner"] == "Independent"
results["rep"] = results["winner"] == "Republican"
seats = results.groupby(["model", "draw"])[["dem", "ind", "rep"]].sum()
seats["dem"] = seats["dem"] + dem_not_up
seats["rep"] = seats["rep"] + rep_not_up

# Republicans control with 50 seats, because the Republican Vice President breaks ties
print("Chance of Republican control (50+ seats), by model:")
print((seats["rep"] >= 50).groupby("model").mean())

print("\nStatus of answers, by model:")
print(pd.crosstab(scores["model"], scores["status"]))


#### Save data ####
nominees.to_csv("data/00-simulated_data/simulated_nominees.csv", index=False)
candidates.to_csv("data/00-simulated_data/simulated_candidates.csv", index=False)
personas.to_csv("data/00-simulated_data/simulated_personas.csv", index=False)
scores.to_csv("data/00-simulated_data/simulated_scores.csv", index=False)
forecast.to_parquet("data/00-simulated_data/simulated_forecast.parquet", index=False)
