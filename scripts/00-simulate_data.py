#### Preamble ####
# Purpose: Simulates the persona sample, the LLM responses, and the forecast
#          for our 2026 US Senate silicon-sampling project.
# Author: Kexin Liu
# Date: 20 October 2026
# Contact: ws1nn2lj3@gmail.com
# License: MIT
# Pre-requisites: pip install pandas numpy pyarrow


#### Workspace setup ####
import os
import numpy as np
import pandas as pd

np.random.seed(2026)


#### Races and candidates ####
# States use two-letter codes, like the CES `state` column.
# (Our real candidates CSV uses full names, so the cleaning script must convert them.)
regular_states = [
    "AL", "AK", "AR", "CO", "DE", "GA", "ID", "IL", "IA", "KS", "KY",
    "LA", "ME", "MA", "MI", "MN", "MS", "MT", "NE", "NH", "NJ", "NM",
    "NC", "OK", "OR", "RI", "SC", "SD", "TN", "TX", "VA", "WV", "WY",]
special_states = ["FL", "OH"]
all_states = regular_states + special_states
 
# Every race has a Republican and a Democrat, with three exceptions that
# copy situations in our real candidates CSV:
#   - NE has an Independent instead of a Democrat
#   - FL adds an Independent
#   - OH adds an Independent and a Libertarian
rows = []
 
for state in all_states:
    race_type = "special" if state in special_states else "regular"
 
    parties = ["Republican", "Democratic"]
    if state == "NE":
        parties = ["Republican", "Independent"]
    if state == "FL":
        parties = ["Republican", "Democratic", "Independent"]
    if state == "OH":
        parties = ["Republican", "Democratic", "Independent", "Libertarian"]
 
    for party in parties:
        rows.append({
            "state": state,
            "race_type": race_type,
            "name": state + " " + party + " candidate", ## Fake names
            "party": party,
        })
 
candidates = pd.DataFrame(rows)
 
 
#### Categories ####
# Labels copied from the CES columns we will use
age_groups = ["18-29", "30-44", "45-64", "65+"]
family_incomes = [
    "Less than $10,000", "$10,000 - $19,999", "$20,000 - $29,999",
    "$30,000 - $39,999", "$40,000 - $49,999", "$50,000 - $59,999",
    "$60,000 - $69,999", "$70,000 - $79,999", "$80,000 - $99,999",
    "$100,000 - $119,999", "$120,000 - $149,999", "$150,000 - $199,999",
    "$200,000 - $249,999", "$250,000 - $349,999", "$350,000 - $499,999",
    "$500,000 or more", "Prefer not to say",]
educations = ["No HS", "HS grad", "Some college", "2-year", "4-year", "Post-grad"]
genders = ["Man", "Woman", "Non-binary", "Other"]
race_groups = ["White", "Black", "Hispanic", "Asian", "Other"]
ideologies = ["Very liberal", "Liberal", "Moderate", "Conservative",
              "Very conservative", "Not sure"]
 
# party_id_7 grouped into three "sides". Leaners go with the party they lean to.
party_labels = {
    "dem":    ["Strong Dem", "Weak Dem", "Lean Dem"],
    "rep":    ["Strong Rep", "Weak Rep", "Lean Rep"],
    "middle": ["Independent", "Not sure"],}
 
# Ideology depends on side, so our fake people are plausible
ideology_probs = {
    "dem":    [0.25, 0.40, 0.25, 0.04, 0.01, 0.05],
    "rep":    [0.01, 0.04, 0.25, 0.40, 0.25, 0.05],
    "middle": [0.05, 0.10, 0.45, 0.10, 0.05, 0.25],}
 
 
#### Simulate personas ####
# 100 people per state. Because we will sample the real CES in proportion
# to vvweight, each persona already counts equally, so there is no weight column.
n_per_state = 100
rows = []
caseid = 1
 
for state in all_states:
    # Give each state its own partisan lean
    p_dem = np.random.uniform(0.30, 0.50)
    p_mid = 0.20
    p_rep = 1 - p_dem - p_mid
 
    for i in range(n_per_state):
        side = np.random.choice(["dem", "rep", "middle"], p=[p_dem, p_rep, p_mid])
 
        rows.append({
            "caseid": caseid,
            "state": state,
            "age_group": np.random.choice(age_groups, p=[0.15, 0.25, 0.35, 0.25]),
            "family_income": np.random.choice(family_incomes),
            "education": np.random.choice(educations, p=[0.05, 0.25, 0.20, 0.10, 0.25, 0.15]),
            "gender": np.random.choice(genders, p=[0.48, 0.50, 0.01, 0.01]),
            "race_5": np.random.choice(race_groups, p=[0.65, 0.12, 0.12, 0.05, 0.06]),
            "party_id_7": np.random.choice(party_labels[side]),
            "ideology": np.random.choice(ideologies, p=ideology_probs[side]),})
        caseid = caseid + 1
 
personas = pd.DataFrame(rows)
 
 
#### Simulate LLM responses ####
models = ["gpt_a", "gpt_b", "claude", "deepseek"]
 
# Share of answers each model fails to give (refusals, timeouts)
na_rates = {"gpt_a": 0.02, "gpt_b": 0.02, "claude": 0.05, "deepseek": 0.03}
 
# How much each side likes each kind of candidate (relative weights).
# "main_left" is the Democrat, or the Independent where there is no Democrat.
weights = {
    "dem":    {"Republican": 5,  "main_left": 88, "third": 2,  "Undecided": 5},
    "rep":    {"Republican": 88, "main_left": 5,  "third": 2,  "Undecided": 5},
    "middle": {"Republican": 35, "main_left": 35, "third": 15, "Undecided": 15},}
 
# Look up which side each party_id_7 label belongs to
side_of = {}
for side in party_labels:
    for label in party_labels[side]:
        side_of[label] = side
 
rows = []
 
for model in models:
    for _, person in personas.iterrows():
        state = person["state"]
        side = side_of[person["party_id_7"]]
 
        # The answers this persona could give: their state's parties, or Undecided
        options = list(candidates[candidates["state"] == state]["party"]) + ["Undecided"]
        has_dem = "Democratic" in options
 
        # Turn the side's weights into probabilities for these options
        probs = []
        for option in options:
            if option == "Republican" or option == "Undecided":
                probs.append(weights[side][option])
            elif option == "Democratic" or (option == "Independent" and not has_dem):
                probs.append(weights[side]["main_left"])
            else:
                probs.append(weights[side]["third"])
        probs = np.array(probs) / sum(probs)
 
        vote = np.random.choice(options, p=probs)
 
        # Plant some problems that our tests should catch
        u = np.random.uniform()
        if u < na_rates[model]:
            vote = None                    # model gave no answer
        elif u < na_rates[model] + 0.01:
            vote = "I'd rather not say"    # model gave an answer we can't use
 
        rows.append({
            "caseid": person["caseid"],
            "state": state,
            "model": model,
            "vote_choice": vote,})
 
responses = pd.DataFrame(rows)
 
 
#### Simulate forecast ####
# For each model and state, resample the 100 answers with replacement
# 1,000 times. Each resample gives one possible result for that race.
# Shares are among answers naming a candidate's party, so NAs, junk and
# "Undecided" are left out, and the shares in each race add up to 1.
n_draws = 1000
rows = []
 
for model in models:
    for state in all_states:
        keep = (responses["model"] == model) & (responses["state"] == state)
        votes = responses[keep]["vote_choice"].to_numpy()
        parties = list(candidates[candidates["state"] == state]["party"])
 
        for draw in range(1, n_draws + 1):
            resample = np.random.choice(votes, size=len(votes), replace=True)
 
            counts = {}
            for party in parties:
                counts[party] = np.sum(resample == party)
            n_valid = sum(counts.values())
 
            # The winner is the party with the most votes
            winner = max(counts, key=counts.get)
 
            for party in parties:
                rows.append({
                    "draw": draw,
                    "state": state,
                    "model": model,
                    "party": party,
                    "share": counts[party] / n_valid,
                    "winner": winner, })
 
forecast = pd.DataFrame(rows)
 
 
#### Quick summary ####
# Seats not up in 2026 (check these before using real data)
dem_not_up = 34
rep_not_up = 31
 
# One row per model, draw and state, then count Democratic wins
results = forecast[["model", "draw", "state", "winner"]].drop_duplicates()
results["dem_win"] = results["winner"] == "Democratic"
dem_seats = results.groupby(["model", "draw"])["dem_win"].sum() + dem_not_up
 
# Democrats need 51 seats for control, because the Vice President breaks ties
print("Chance of Democratic control, by model:")
print((dem_seats >= 51).groupby("model").mean())
 
print("\nChance each party wins each state (first model):")
first = results[results["model"] == models[0]]
print(pd.crosstab(first["state"], first["winner"], normalize="index").head(10))
 
 
#### Save data ####
candidates.to_csv("data/00-simulated_data/simulated_candidates.csv", index=False)
personas.to_csv("data/00-simulated_data/simulated_personas.csv", index=False)
responses.to_csv("data/00-simulated_data/simulated_responses.csv", index=False)
forecast.to_parquet("data/00-simulated_data/simulated_forecast.parquet", index=False)
