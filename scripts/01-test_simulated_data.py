#### Preamble ####
# Purpose: Test suite for the SIMULATED races, candidates, personas, scored answers and forecast
#          (made by scripts/00-simulate_data.py), written with pytest (the standard Python test
#          framework, the counterpart of R's testthat) and pandera (data validation, the
#          counterpart of validate and pointblank). The real data have their own suite in
#          scripts/12-test_actual_data.py; both use the schemas in scripts/schemas.py.
# Author: Kexin (Coco) Liu
# Date: 27 October 2026 (rewritten from the assert version of 25 October)
# Contact: ws1nn2lj3@gmail.com
# License: MIT
# Pre-requisites: Run scripts/00-simulate_data.py first. pip install pytest pandera.
# Any other information needed? Run from the project root, either way:
#   python scripts/01-test_simulated_data.py
#   python -m pytest scripts/01-test_simulated_data.py -v      (one line per test)
#   Every test has a docstring saying what it checks and why. Each table is first checked against
#   its pandera schema (types, allowed values, ranges, keys); the tests after that check the
#   relations between tables, which a schema of one table cannot see. Tests that run once per
#   model are parametrized, so a failure names the model.
#
# HOW the tests are organized
#   1. Schemas         one test per table, using scripts/schemas.py
#   2. Races           35 races, two candidates each, long and wide files agree
#   3. Personas        100 per state, repeated CES respondents consistent, weight dropped
#   4. Scored answers  complete grid, NAs only where a model did not answer, labels match
#   5. Forecast        shares add to 1, the winner has the larger share, 100 seats per draw

#### Workspace setup ####
import sys

import numpy as np
import pandas as pd
import pytest

import config
import schemas as S

SIM = "data/00-simulated_data/"
N_RACES, N_PER_STATE, N_DRAWS = 35, 100, 1000
MODELS = list(config.MODELS)  # haiku, luna, deepseek, gemini
N_REPEATS = 3
DEM_NOT_UP, REP_NOT_UP = 34, 31  # seats not up in 2026, as in script 11
MAX_PROBLEM_RATE = 0.10  # largest share of unscorable answers accepted per model
MIN_SCORED_PERSONAS = 50  # fewest scored personas accepted in a state for one model
KEYS = ["draw", "state", "model"]


#### Data (read once for the whole suite) ####
@pytest.fixture(scope="module")
def nominees():
    return pd.read_csv(SIM + "simulated_nominees.csv")


@pytest.fixture(scope="module")
def candidates():
    return pd.read_csv(SIM + "simulated_candidates.csv")


@pytest.fixture(scope="module")
def personas():
    return pd.read_csv(SIM + "simulated_personas.csv")


@pytest.fixture(scope="module")
def scores():
    return pd.read_csv(SIM + "simulated_scores.csv")


@pytest.fixture(scope="module")
def forecast():
    return pd.read_parquet(SIM + "simulated_forecast.parquet")


@pytest.fixture(scope="module")
def left_party_of(nominees):
    """The left party of each state: Democratic, or Independent where there is no Democrat."""
    return dict(zip(nominees["state"], nominees["left_party"]))


#### 1. Schemas ####
def test_schema_nominees(nominees):
    """Races: 35 known states, one row each, two different names, allowed race types and parties."""
    S.validate(S.nominees_schema(list(config.STATE_ABBR.values()), N_RACES), nominees)


def test_schema_candidates(candidates, nominees):
    """Long candidate file: allowed parties, no candidate listed twice, one Republican per race."""
    S.validate(S.candidates_schema(list(nominees["state"])), candidates)


def test_schema_personas(personas, nominees):
    """Personas: unique ids, only allowed labels, 100 per state, no CES weight left behind."""
    S.validate(S.personas_schema(list(nominees["state"]), N_PER_STATE), personas)


def test_schema_scores(scores, personas):
    """Scored answers: known models, repeats and statuses; probabilities only for ok answers, in
    0 to 1 and adding up to 1."""
    S.validate(S.scores_schema(MODELS, N_REPEATS, list(personas["persona_id"])), scores)


def test_schema_forecast(forecast, nominees):
    """Forecast: draws 1 to 1,000, shares in 0 to 1 that add up to 1, one winner per race and draw."""
    S.validate(S.forecast_schema(list(nominees["state"]), MODELS, N_DRAWS), forecast)


#### 2. Races ####
def test_two_candidates_per_race(candidates):
    """The long file has exactly two rows per race: the left candidate and the Republican."""
    assert candidates.groupby("state").size().eq(2).all(), "A race does not have exactly 2 candidates"


def test_long_and_wide_files_agree(candidates, left_party_of):
    """The left candidate's party in the long file is the left party in the wide nominee file."""
    left = candidates[candidates["party"] != "Republican"]
    assert (left["party"] == left["state"].map(left_party_of)).all(), "Long and wide files disagree"


def test_both_files_cover_the_same_races(candidates, nominees):
    """No race is in one candidate file and missing from the other."""
    assert set(candidates["state"]) == set(nominees["state"])


#### 3. Personas ####
def test_personas_cover_every_race(personas, nominees):
    """Every race state has personas, and no persona lives in a state without a race."""
    assert set(personas["state"]) == set(nominees["state"])


def test_repeated_respondent_stays_in_one_state(personas):
    """Sampling is with replacement within a state, so a CES respondent can repeat, but only
    inside one state."""
    assert personas.groupby("caseid")["state"].nunique().eq(1).all(), "A caseid appears in two states"


@pytest.mark.parametrize("column", list(S.PERSONA_LABELS))
def test_repeated_respondent_is_identical(personas, column):
    """A respondent drawn twice must have the same value of every persona variable both times."""
    assert personas.groupby("caseid")[column].nunique().eq(1).all(), f"A repeated caseid differs in {column}"


def test_no_missing_persona_values(personas):
    """None of the seven persona variables is missing, so every prompt is complete."""
    assert personas[list(S.PERSONA_LABELS)].notna().all().all()


#### 4. Scored answers ####
def test_planned_models_and_repeats(scores):
    """Exactly the planned four models and three repeats are present."""
    assert sorted(scores["model"].unique()) == sorted(MODELS)
    assert sorted(scores["repeat"].unique()) == list(range(1, N_REPEATS + 1))


def test_complete_grid(scores, personas):
    """Every persona answered once per model and repeat: nothing missing, nothing twice."""
    assert len(scores) == len(personas) * len(MODELS) * N_REPEATS, "Wrong number of answers"
    assert set(scores["persona_id"]) == set(personas["persona_id"])


def test_vote_choice_matches_probabilities(scores, left_party_of):
    """The readable label is the option with the largest probability, named with that race's
    parties; an answer that is not ok has no label."""
    ok = scores[scores["status"] == "ok"]
    labels = np.array([[left_party_of[s], "Republican", "Undecided"] for s in ok["state"]])
    expected = labels[np.arange(len(ok)), ok[S.P_COLS].to_numpy(dtype=float).argmax(axis=1)]
    assert (ok["vote_choice"].to_numpy() == expected).all(), "vote_choice does not match the probabilities"
    assert scores.loc[scores["status"] != "ok", "vote_choice"].isna().all()


@pytest.mark.parametrize("model", MODELS)
def test_problem_rate_below_limit(scores, model):
    """Each model's share of answers that cannot be scored (refusals, empty, errors) is below 10%."""
    rate = (scores.loc[scores["model"] == model, "status"] != "ok").mean()
    assert rate < MAX_PROBLEM_RATE, f"{model}: {rate:.1%} of answers cannot be scored"


@pytest.mark.parametrize("model", MODELS)
def test_enough_scored_personas_in_every_state(scores, nominees, model):
    """Every state has at least 50 scored personas for the model, so no state share rests on a
    handful of answers."""
    ok = scores[(scores["status"] == "ok") & (scores["model"] == model)]
    counts = ok.groupby("state")["persona_id"].nunique().reindex(nominees["state"], fill_value=0)
    assert counts.ge(MIN_SCORED_PERSONAS).all(), (
        f"{model}: too few scored personas in {list(counts[counts < 50].index)}"
    )


#### 5. Forecast ####
def test_share_is_for_a_party_in_the_race(forecast, left_party_of):
    """Each share belongs to the state's left party or the Republican, never to a party not running."""
    expected = np.where(forecast["party"] == "Republican", "Republican", forecast["state"].map(left_party_of))
    assert (forecast["party"] == expected).all()


def test_winner_has_the_larger_share(forecast):
    """The winner is the party with the larger share (exact ties are skipped: script 11 breaks
    them with a coin flip)."""
    untied = forecast[forecast.groupby(KEYS)["share"].transform("nunique") > 1]
    top = untied.loc[untied.groupby(KEYS)["share"].idxmax()]
    assert (top["party"] == top["winner"]).all(), "Winner is not the party with the larger share"


def test_every_state_and_model_has_all_draws(forecast):
    """Every state and model pair has all 1,000 draws, so seat counts are not made from partial draws."""
    draws = forecast.groupby(["state", "model"])["draw"].nunique()
    assert len(draws) == N_RACES * len(MODELS) and draws.eq(N_DRAWS).all()


def test_every_draw_fills_the_senate(forecast):
    """With the 65 seats not up, every draw adds up to exactly 100 senators."""
    races = forecast[["model", "draw", "state"]].drop_duplicates().groupby(["model", "draw"]).size()
    assert (races + DEM_NOT_UP + REP_NOT_UP).eq(100).all(), "Seats do not add up to 100"


#### Run as a script ####
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-rfE", "-p", "no:cacheprovider"]))
