#### Preamble ####
# Purpose: Test suite for the REAL candidates, personas, prompts, model answers (the API
#          responses), scores, forecast and combination with polls of one run, written with
#          pytest (the standard Python test framework, the counterpart of R's testthat) and
#          pandera (data validation, the counterpart of validate and pointblank). The simulated
#          data have their own suite in scripts/01-test_simulated_data.py; both use the schemas in
#          scripts/schemas.py, so the simulated and the real data meet the same rules.
# Author: Kexin (Coco) Liu, from the group's assert version of 22 October
# Date: 27 October 2026
# License: MIT
# Pre-requisites: Run scripts 06 to 11 for this RUN (and 16 for the crosstab tests).
#   pip install pytest pandera (in requirements.txt).
# Any other information needed? Run from the project root, either way:
#   python scripts/12-test_actual_data.py                    (what GitHub Actions runs)
#   python -m pytest scripts/12-test_actual_data.py -v       (one line per test)
#   RUN selects the run, as in every script (export RUN=2026-10-27, or RUN=nonews).
#   Every test has a docstring saying what it checks and why. If one fails, fix the data or the
#   earlier script, not the test. In the workflow a failure stops the forecast and the website
#   from being published, while the paid answers are still saved.
#
# HOW the tests are organized
#   1. Schemas          one test per table, using scripts/schemas.py
#   2. Candidates and personas
#   3. Prompts          one per persona, from this run, naming both candidates
#   4. API responses    complete, from the requested model, not truncated, with tokens and times
#   5. Scores           every call scored, every state scored by every model
#   6. Forecast         the winner has the larger share, 100 seats per draw
#   7. Combination with polls (script 11, --combine ivw) and the chamber summary
#   8. Crosstabs (script 16)

#### Workspace setup ####
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import config
import schemas as S

N_RACES = 35
DEM_NOT_UP, REP_NOT_UP = 34, 31          # seats not up in 2026, as in script 11
MODELS = list(config.MODELS)
MAX_PROBLEM_RATE = 0.10                  # largest share of empty or failed answers accepted per model
MAX_TRUNCATED_RATE = 0.01                # largest share of answers cut off at the token limit, per model
MAX_SHORT_RATE = 0.01                    # largest share of scored answers under 20 characters, per model
KEYS = ["draw", "state", "model"]
CALL = ["prompt_id", "model", "repeat"]   # identifies one API call


#### Data (read once for the whole suite) ####
@pytest.fixture(scope="module")
def nominees():
    return pd.read_csv(config.NOMINEES_FILE)


@pytest.fixture(scope="module")
def personas():
    return config.read_table(config.PERSONAS_FILE)


@pytest.fixture(scope="module")
def prompts():
    return config.read_table(config.PROMPTS_FILE)


@pytest.fixture(scope="module")
def responses():
    r = config.read_table(config.RESPONSES_FILE)
    r["error"] = r["error"].fillna("")
    r["response_text"] = r["response_text"].fillna("")
    return r


@pytest.fixture(scope="module")
def scores():
    return config.read_table(config.SCORES_FILE)


@pytest.fixture(scope="module")
def scored_text(scores, responses):
    """Each scored answer with its text."""
    return scores[CALL + ["status"]].merge(responses[CALL + ["response_text"]], on=CALL)


@pytest.fixture(scope="module")
def forecast():
    return config.read_table(config.FORECAST_FILE)


@pytest.fixture(scope="module")
def n_per_state(personas):
    return int(personas.groupby("state").size().iloc[0])


@pytest.fixture(scope="module")
def n_repeats(responses):
    return int(responses["repeat"].nunique())


def answered(responses, model):
    """One model's calls that came back without an error."""
    return responses[(responses["error"] == "") & (responses["model"] == model)]


#### 1. Schemas ####
def test_schema_nominees(nominees):
    """Races: 35 known states, one row each, two different names, 2 special elections."""
    S.validate(S.nominees_schema(list(config.STATE_ABBR.values()), N_RACES), nominees)


def test_schema_personas(personas, nominees, n_per_state):
    """Personas: unique ids, only allowed labels, the same number in every state, a persona text,
    and no CES weight left behind (so nobody is weighted twice)."""
    S.validate(S.personas_schema(list(nominees["state"]), n_per_state, with_text=True), personas)


def test_schema_prompts(prompts, personas):
    """Prompts: unique ids, all from this run, with or without news as the run requires."""
    schema = S.pa.DataFrameSchema({
        "prompt_id": S.pa.Column(str, S.pa.Check.str_endswith("|" + config.RUN), unique=True),
        "run": S.pa.Column(str, S.pa.Check.eq(config.RUN)),
        "persona_id": S.pa.Column(str, S.pa.Check.isin(list(personas["persona_id"]))),
        "news": S.pa.Column(bool, S.pa.Check.eq(config.NEWS_RUN)),
        "prompt_text": S.pa.Column(str, S.pa.Check.str_length(min_value=100)),
    })
    S.validate(schema, prompts)


def test_schema_responses(responses, personas, n_repeats):
    """API responses: one row per prompt, model and repeat; known models; this run only;
    non-negative token counts; a query time in ISO format."""
    schema = S.pa.DataFrameSchema({
        "prompt_id": S.pa.Column(str, S.pa.Check.str_endswith("|" + config.RUN)),
        "run": S.pa.Column(str, S.pa.Check.eq(config.RUN)),
        "persona_id": S.pa.Column(str, S.pa.Check.isin(list(personas["persona_id"]))),
        "model": S.pa.Column(str, S.pa.Check.isin(MODELS)),
        "repeat": S.pa.Column(int, S.pa.Check.in_range(1, n_repeats)),
        "input_tokens": S.pa.Column(float, S.pa.Check.ge(0), nullable=True, coerce=True),
        "output_tokens": S.pa.Column(float, S.pa.Check.ge(0), nullable=True, coerce=True),
        "queried_at_utc": S.pa.Column(str, S.pa.Check.str_matches(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")),
    }, unique=CALL)
    S.validate(schema, responses)


def test_schema_scores(scores, personas, n_repeats):
    """Scores: probabilities only for answers with status ok, between 0 and 1, adding up to 1.
    This is the rule that stops a model that did not answer (an NA) from reaching the forecast."""
    S.validate(S.scores_schema(MODELS, n_repeats, list(personas["persona_id"])), scores)


def test_schema_forecast(forecast, nominees):
    """Forecast: no missing shares, shares between 0 and 1 adding up to 1 (never above 100%),
    two rows and one winner per race, draw and model."""
    S.validate(S.forecast_schema(list(nominees["state"]), MODELS), forecast)


#### 2. Candidates and personas ####
def test_personas_cover_every_race(personas, nominees):
    """Every race state has personas, and no persona lives in a state without a race."""
    assert set(personas["state"]) == set(nominees["state"])


def test_repeated_respondent_stays_in_one_state(personas):
    """Sampling is with replacement within a state: a CES respondent may repeat, but in one state only."""
    assert personas.groupby("caseid")["state"].nunique().eq(1).all(), "A caseid appears in two states"


@pytest.mark.parametrize("column", list(S.PERSONA_LABELS))
def test_repeated_respondent_is_identical(personas, column):
    """A respondent drawn twice has the same value of every persona variable both times."""
    assert personas.groupby("caseid")[column].nunique().eq(1).all(), f"A repeated caseid differs in {column}"


def test_presidential_vote_not_in_persona_text(personas):
    """The 2024 presidential vote is kept out of the prompt (it is used to check the answers), so
    the persona text must not mention it."""
    assert not personas["persona_text"].str.contains("Trump|Harris|president", case=False).any()


#### 3. Prompts ####
def test_one_prompt_per_persona(prompts, personas):
    """Every persona has a prompt and every prompt belongs to a persona."""
    assert set(prompts["persona_id"]) == set(personas["persona_id"])


def test_prompt_names_both_candidates(prompts, nominees):
    """Each prompt names both candidates of its own state's race, so no persona is asked about
    the wrong race."""
    m = prompts.merge(nominees, on="state")
    ok = [a in t and b in t for a, b, t in zip(m["left_name"], m["rep_name"], m["prompt_text"])]
    assert all(ok), f"{len(ok) - sum(ok)} prompts do not name both candidates"


#### 4. API responses ####
def test_complete_grid(responses, prompts, n_repeats):
    """Every prompt was sent to every model the planned number of times, once each."""
    assert n_repeats == config.N_REPEATS, f"{n_repeats} repeats, config says {config.N_REPEATS}"
    assert sorted(responses["model"].unique()) == sorted(MODELS), "Wrong set of models"
    assert len(responses) == len(prompts) * len(MODELS) * n_repeats, "Wrong number of responses"
    assert set(responses["prompt_id"]) <= set(prompts["prompt_id"]), "An answer has no matching prompt"


@pytest.mark.parametrize("model", MODELS)
def test_problem_rate_below_limit(responses, model):
    """Each model's share of empty or failed calls is below 10%."""
    r = responses[responses["model"] == model]
    rate = ((r["response_text"].str.strip() == "") | (r["error"] != "")).mean()
    assert rate < MAX_PROBLEM_RATE, f"{model}: {rate:.1%} of calls empty or failed"


@pytest.mark.parametrize("model", MODELS)
def test_requested_model_is_the_configured_one(responses, model):
    """Every call asked OpenRouter for the model id in config.MODELS."""
    requested = responses.loc[responses["model"] == model, "model_requested"]
    assert requested.eq(config.MODELS[model]).all(), f"{model}: requested {sorted(set(requested))}"


@pytest.mark.parametrize("model", MODELS)
def test_answer_came_from_the_requested_provider(responses, model):
    """The model that answered comes from the company requested (the part of the id before "/").
    If the exact id changed, the test passes with a warning, because a provider can update the
    model behind a name (Bisbee et al. 2024)."""
    r = answered(responses, model)
    provider = config.MODELS[model].split("/")[0]
    assert r["model_returned"].fillna("").str.startswith(provider + "/").all(), f"{model}: another provider answered"
    other = sorted(set(r["model_returned"]) - {config.MODELS[model]})
    if other:
        warnings.warn(f"{model}: answers also came from {other}")


@pytest.mark.parametrize("model", MODELS)
def test_few_answers_cut_off(responses, model):
    """Answers that stopped at the token limit (finish_reason "length") may be cut off
    mid-sentence; fewer than 1% per model."""
    rate = responses.loc[responses["model"] == model, "finish_reason"].eq("length").mean()
    assert rate < MAX_TRUNCATED_RATE, f"{model}: {rate:.1%} of answers cut off"


@pytest.mark.parametrize("model", MODELS)
def test_tokens_recorded_for_answered_calls(responses, model):
    """Every answered call has a positive input token count and an output count, which the cost
    table (script 16d) relies on."""
    r = answered(responses, model)
    assert r["input_tokens"].gt(0).all() and r["output_tokens"].notna().all(), f"{model}: token counts missing"


def test_queried_during_this_run(responses):
    """Answers were collected within three days and, for a news day, not before that day, so
    answers from another run cannot be mixed in."""
    t = pd.to_datetime(responses["queried_at_utc"], utc=True)
    assert t.max() - t.min() < pd.Timedelta(days=3), "Answers span more than three days"
    if config.RUN != "nonews":
        assert t.min().date() >= pd.Timestamp(config.RUN).date(), "An answer is older than the run"


#### 5. Scores ####
def test_every_response_is_scored(scores, responses):
    """Scoring kept every call: a failed call gets a status, it is not dropped."""
    assert len(scores) == len(responses)
    assert scores[CALL].merge(responses[CALL], on=CALL).shape[0] == len(responses), "Scores and calls do not match"


def test_scored_answers_have_text(scored_text):
    """An answer counted as ok is never empty, so SSR always scored real text."""
    assert (scored_text.loc[scored_text["status"] == "ok", "response_text"].str.strip() != "").all()


@pytest.mark.parametrize("model", MODELS)
def test_few_very_short_answers(scored_text, model):
    """Fewer than 1% of a model's scored answers are under 20 characters: the prompt asks for two
    or three sentences, and a very short answer is close to a refusal."""
    ok = scored_text[(scored_text["status"] == "ok") & (scored_text["model"] == model)]
    rate = ok["response_text"].str.strip().str.len().lt(20).mean()
    assert rate < MAX_SHORT_RATE, f"{model}: {rate:.1%} of scored answers under 20 characters"


@pytest.mark.parametrize("model", MODELS)
def test_every_state_scored(scores, nominees, model):
    """Every state has at least one scored persona for the model, so no state share is missing."""
    counts = scores[(scores["model"] == model) & scores["p_left"].notna()].groupby("state").size()
    missing = sorted(set(nominees["state"]) - set(counts.index))
    assert not missing, f"{model}: no scored answers in {missing}"


#### 6. Forecast ####
def test_winner_has_the_larger_share(forecast):
    """The winner is the party with the larger share (exact 50/50 ties are skipped: script 11
    breaks them with a coin flip)."""
    untied = forecast[forecast.groupby(KEYS)["share"].transform("nunique") > 1]
    top = untied.loc[untied.groupby(KEYS)["share"].idxmax()]
    assert (top["party"] == top["winner"]).all()


def test_winner_is_running_in_that_race(forecast, nominees):
    """The winner is the state's left party or the Republican, never a party not on the ballot."""
    left = nominees.set_index("state")["left_party"]
    w = forecast[["state", "winner"]].drop_duplicates()
    assert ((w["winner"] == "Republican") | (w["winner"] == w["state"].map(left))).all()


def test_same_number_of_draws_everywhere(forecast):
    """Every state and model has the same number of draws, so seat counts use complete draws."""
    draws = forecast.groupby(["state", "model"])["draw"].nunique()
    assert len(draws) == N_RACES * len(MODELS) and draws.nunique() == 1


def test_every_draw_fills_the_senate(forecast):
    """With the 65 seats not up, every draw adds up to exactly 100 senators."""
    races = forecast[["model", "draw", "state"]].drop_duplicates().groupby(["model", "draw"]).size()
    assert (races + DEM_NOT_UP + REP_NOT_UP).eq(100).all(), "Seats do not add up to 100"


#### 7. Combination with polls and the chamber summary ####
@pytest.fixture(scope="module")
def ivw():
    path = config.RUN_RESULTS / "ivw_weights.csv"
    if not path.exists():
        pytest.skip("no ivw_weights.csv; run scripts/11-build_forecast.py without --combine")
    return pd.read_csv(path)


def test_ivw_file_exists_when_used():
    """If the headline forecast was built with --combine ivw, its weights file must be saved."""
    chamber = config.RUN_RESULTS / "chamber_summary.csv"
    if chamber.exists() and "ivw" in set(pd.read_csv(chamber).get("combine", [])):
        assert (config.RUN_RESULTS / "ivw_weights.csv").exists()


def test_schema_ivw_weights(ivw, nominees):
    """Inverse-variance weights: one row per state and model, no missing input, weights and shares
    between 0 and 1, positive variances, and one of two outside estimates."""
    schema = S.pa.DataFrameSchema({
        "model": S.pa.Column(str, S.pa.Check.isin(MODELS)),
        "state": S.pa.Column(str, S.pa.Check.isin(list(nominees["state"]))),
        "source": S.pa.Column(str, S.pa.Check.isin(["poll", "2024 + swing"])),
        "m_star": S.pa.Column(float),
        "t": S.pa.Column(float),
        "combined": S.pa.Column(float, S.prob),
        "w_synthetic": S.pa.Column(float, S.prob),
        "var_m": S.pa.Column(float, S.pa.Check.gt(0)),
        "var_t": S.pa.Column(float, S.pa.Check.gt(0)),
    }, unique=["model", "state"], checks=S.pa.Check(lambda d: len(d) == N_RACES * len(MODELS),
                                                    error="a state and model pair has no weight"))
    S.validate(schema, ivw)


def test_combined_share_between_its_inputs(ivw):
    """A weighted average lies between the two things it averages: the bias-corrected synthetic
    share and the outside estimate."""
    lo = np.minimum(ivw["m_star"], ivw["t"]) - 1e-6
    hi = np.maximum(ivw["m_star"], ivw["t"]) + 1e-6
    assert ivw["combined"].between(lo, hi).all()


def test_models_agree_on_which_states_are_polled(ivw):
    """Whether a state has polls does not depend on the model."""
    assert ivw.groupby("state")["source"].nunique().eq(1).all()


def test_chamber_summary():
    """Chamber summary: one row per model, chances between 0 and 1, and Republican control and a
    51-seat Democratic majority together in no more than 100% of draws."""
    path = config.RUN_RESULTS / "chamber_summary.csv"
    if not path.exists():
        pytest.skip("no chamber_summary.csv")
    schema = S.pa.DataFrameSchema({
        "model": S.pa.Column(str, S.pa.Check.isin(MODELS), unique=True),
        "dem_median": S.pa.Column(float, S.pa.Check.in_range(0, 100), coerce=True),
        "rep_median": S.pa.Column(float, S.pa.Check.in_range(0, 100), coerce=True),
        "p_rep_control": S.pa.Column(float, S.prob),
        "p_dem_51_without_ind": S.pa.Column(float, S.prob),
    }, checks=S.pa.Check(lambda d: (d["p_rep_control"] + d["p_dem_51_without_ind"]).le(1 + 1e-9).all(),
                         error="the two chances add up to more than 1"))
    S.validate(schema, pd.read_csv(path))


#### 8. Crosstabs ####
@pytest.fixture(scope="module")
def cross():
    path = config.RUN_RESULTS / "crosstabs.csv"
    if not path.exists():
        pytest.skip("no crosstabs.csv; run scripts/16-make_crosstabs.py")
    return pd.read_csv(path)


def test_crosstab_shares(cross):
    """Crosstab shares are present and between 0 and 1."""
    each = cross[cross["model"] != "mean of models"]
    assert each[["share_2026", "lean_left"]].notna().all().all()
    assert each[["share_2026", "lean_left"]].stack().between(0, 1).all()


def test_crosstab_groups_complete(cross, personas):
    """Every model is in every cell, and each variable's groups add up to all the personas."""
    each = cross[cross["model"] != "mean of models"]
    mean = cross[cross["model"] == "mean of models"]
    assert each.groupby(["variable", "category"])["model"].nunique().eq(len(MODELS)).all()
    assert mean.groupby("variable")["n_personas"].sum().eq(len(personas)).all()


#### Run as a script ####
if __name__ == "__main__":
    print(f"Testing run {config.RUN}")
    sys.exit(pytest.main([__file__, "-q", "-rfEw", "-p", "no:cacheprovider"]))
