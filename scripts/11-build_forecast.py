#### Preamble ####
# Purpose: Turns persona-level probabilities into a distribution of outcomes for each state and
#          model, saves it in the same layout as the simulated forecast, and builds a table
#          that sets the model's share beside the poll average and the 2024 presidential vote.
# Author: (group)
# Date: 22 October 2026
# License: MIT
# Pre-requisites: Run scripts/10-score_responses.py first. Needs pyarrow for the parquet file.
# Any other information needed? Run from the project root.
#   python scripts/11-build_forecast.py                 <- model only (test-run default)
#   python scripts/11-build_forecast.py --calibrate     <- also subtract the average gap to polls
#
# HOW one number per state is built:
#   1. Average each persona's probabilities over repeats (one row per persona and model).
#   2. A state's Democratic-side share = mean p_left / (mean p_left + mean p_rep).
#      Because personas were drawn in proportion to the CES weights (script 07), every persona
#      counts the same: this IS the poststratification step for the "digital twin" design.
# HOW the distribution is built (matches the simulated forecast script): resample the
#   personas of a state with replacement N_DRAWS times and recompute the share each time.
#   This captures persona-sampling error only. With 5 personas per state the spread is huge
#   (a share near 50% has a standard error of about 22 points), so the test-run forecast is a
#   check that the code works, NOT a forecast. With 100 personas it is about 5 points.
# A national error shared by all states is ON by default (config.NATIONAL_SD = 0.03, an
#   assumption; Gelman et al. 2020; Shirani-Mehr et al. 2018). Rerun with --national-sd 0, 0.02
#   and 0.04 for the sensitivity table.
# NOT included yet (say so in the paper): spread between models and repeats, and error in the
#   poll correction.
# HOW seats are counted: Nebraska, Idaho and South Dakota have an independent instead of a
#   Democrat, and Osborn (NE) has said he will not caucus with either party, so independent wins
#   are counted as their own category, not as Democratic seats. The headline is the chance that
#   Republicans keep control: with Vice President Vance breaking 50-50 ties, Republicans control
#   the Senate with 50 or more seats, whichever side the independents join.
# WHY polls are used only as a check or a correction: the model is biased toward one side
# (Pew 2026; Heath & Alexander 2026). Polls are not the truth either, because poll errors are
# correlated across polls and offices (Shirani-Mehr et al. 2018).
# LIMIT: the correction assumes the model errs by the same amount in every state.


#### Workspace setup ####
import argparse

import numpy as np
import pandas as pd

import config

ap = argparse.ArgumentParser()
ap.add_argument("--design", default="B", choices=["B", "C"], help="which prompt design to forecast from")
ap.add_argument("--draws", type=int, default=config.N_DRAWS)
ap.add_argument("--calibrate", action="store_true", help="subtract the shrunk average model-minus-poll gap")
ap.add_argument("--national-sd", type=float, default=config.NATIONAL_SD,
                help="SD of an error shared by all states (default in config.py; 0 = states independent)")
ap.add_argument("--poll-window-days", type=int, default=45)
args = ap.parse_args()

# Seats not up in 2026, checked 2026-10-25: the Senate is 53 R - 47 D (the 47 include King and
# Sanders, independents who caucus with Democrats). 35 seats are up: 13 held by Democrats and 22
# by Republicans (including the FL and OH specials). 47 - 13 = 34 and 53 - 22 = 31.
DEM_NOT_UP, REP_NOT_UP = 34, 31

rng = np.random.default_rng(config.SEED)
nominees = pd.read_csv(config.NOMINEES_FILE).set_index("state")
personas = config.read_table(config.PERSONAS_FILE)

#### Persona-level probabilities ####
scores = config.read_table(config.SCORES_FILE)
scores = scores[(scores["design"] == args.design) & scores["p_left"].notna()]
per_persona = (scores.groupby(["state", "persona_id", "model"])[["p_left", "p_rep", "p_other"]]
               .mean().reset_index())
n_scored = per_persona.groupby(["state", "model"]).size()
print(f"Scored personas per state and model: min {n_scored.min()}, max {n_scored.max()}")

#### Poll averages ####
def poll_averages():
    """Democratic-side two-party share per state from polls that name BOTH of our nominees.
    Uses the last `poll_window_days` before the newest poll in the file; one poll counted once
    (likely-voter version preferred); weighted by sample size."""
    if not config.POLLS_FILE.exists():
        return pd.DataFrame(columns=["state", "poll_share", "n_polls"])
    polls = pd.read_csv(config.POLLS_FILE, encoding="utf-8-sig")
    polls["end"] = pd.to_datetime(polls["end_date"])
    polls = polls[polls["end"] >= polls["end"].max() - pd.Timedelta(days=args.poll_window_days)]
    polls["ln"] = polls["response_name"].map(config.last_name)
    rows = []
    for state, race in nominees.iterrows():
        sub = polls[polls["state"] == state]
        left = sub[sub["ln"] == config.last_name(race["left_name"])]
        right = sub[sub["ln"] == config.last_name(race["rep_name"])]
        # A name can appear twice in one question (for example 'Dan Sullivan' and 'Dan J. Sullivan');
        # the real candidate is the one with the larger percentage.
        lq = left.groupby("question_id").agg(pct_l=("pct", "max"), poll_id=("poll_id", "first"),
                                             population=("population", "first"),
                                             sample_size=("sample_size", "first"))
        rq = right.groupby("question_id").agg(pct_r=("pct", "max"))
        both = lq.join(rq, how="inner").reset_index()
        if both.empty:
            continue
        both["order"] = both["population"].map({"lv": 0, "rv": 1}).fillna(2)
        both = both.sort_values("order").drop_duplicates("poll_id")
        share = both["pct_l"] / (both["pct_l"] + both["pct_r"])
        weight = both["sample_size"].fillna(600)
        rows.append({"state": state, "poll_share": float(np.average(share, weights=weight)),
                     "n_polls": len(both)})
    return pd.DataFrame(rows)


polls_df = poll_averages()

#### Draws ####
D = args.draws
rows, point = [], []
for model in sorted(per_persona["model"].unique()):
    shift = rng.normal(0, args.national_sd, D) if args.national_sd > 0 else np.zeros(D)
    for state in nominees.index:
        sub = per_persona[(per_persona["model"] == model) & (per_persona["state"] == state)]
        if sub.empty:
            continue
        left, rep = sub["p_left"].to_numpy(), sub["p_rep"].to_numpy()
        idx = rng.integers(0, len(sub), size=(D, len(sub)))
        ml, mr = left[idx].mean(axis=1), rep[idx].mean(axis=1)
        share = np.where(ml + mr > 0, ml / np.where(ml + mr > 0, ml + mr, 1), 0.5)
        point.append({"model": model, "state": state, "n_personas": len(sub),
                      "model_share": float(left.mean() / (left.mean() + rep.mean())),
                      "raw_draws": share, "shift": shift})

# Optional correction toward polls
gap_table = pd.DataFrame()
if args.calibrate:
    pt = pd.DataFrame([{k: v for k, v in p.items() if k not in ("raw_draws", "shift")} for p in point])
    pt = pt.merge(polls_df, on="state", how="left")
    for model, grp in pt.groupby("model"):
        polled = grp.dropna(subset=["poll_share"])
        if len(polled) < 8:
            print(f"--calibrate skipped for {model}: only {len(polled)} polled states (need 8)")
            continue
        gap = polled["model_share"] - polled["poll_share"]
        k = 3.0                                    # placeholder shrinkage strength; try 2 and 5
        lam = polled["n_polls"] / (polled["n_polls"] + k)
        shrunk = lam * gap + (1 - lam) * gap.mean()
        gaps = dict(zip(polled["state"], shrunk))
        mean_gap = float(gap.mean())
        for p in point:
            if p["model"] == model:
                p["gap"] = gaps.get(p["state"], mean_gap)
        gap_table = pd.concat([gap_table, pd.DataFrame({"model": model, "state": polled["state"], "gap": shrunk})])

out = []
for p in point:
    share = np.clip(p["raw_draws"] - p.get("gap", 0.0) + p["shift"], 0, 1)
    race = nominees.loc[p["state"]]
    # An exact 50/50 tie (possible with very few personas) is decided by a coin flip
    win_left = np.where(share == 0.5, rng.random(len(share)) < 0.5, share > 0.5)
    n = len(share)
    base = {"state": p["state"], "model": p["model"]}
    out.append(pd.DataFrame({**base, "draw": np.arange(1, n + 1), "party": race["left_party"],
                             "share": share, "winner": np.where(win_left, race["left_party"], "Republican")}))
    out.append(pd.DataFrame({**base, "draw": np.arange(1, n + 1), "party": "Republican",
                             "share": 1 - share, "winner": np.where(win_left, race["left_party"], "Republican")}))
forecast = pd.concat(out, ignore_index=True)
forecast = forecast[["draw", "state", "model", "party", "share", "winner"]]

#### Comparison table ####
ces = pd.read_csv(config.CES_FILE, low_memory=False,
                  usecols=["state", "vvweight_post", "pres_vote_2024"])
ces = ces[ces["state"].isin(nominees.index) & ces["vvweight_post"].notna()
          & ces["pres_vote_2024"].isin(["Harris", "Trump"])]
w = ces.pivot_table(index="state", columns="pres_vote_2024", values="vvweight_post", aggfunc="sum")
pres = (w["Harris"] / (w["Harris"] + w["Trump"])).rename("pres2024_dem_share").reset_index()

means = forecast[forecast["party"] != "Republican"].groupby(["state", "model"])["share"].mean().rename("forecast_mean")
comparison = (means.reset_index().pivot(index="state", columns="model", values="forecast_mean").add_prefix("model_")
              .reset_index().merge(polls_df, on="state", how="left").merge(pres, on="state", how="left")
              .merge(personas.groupby("state")[["n_pool", "n_eff"]].first().reset_index(), on="state"))

#### Chamber summary ####
results = forecast[["model", "draw", "state", "winner"]].drop_duplicates()
seats = (results.assign(dem=results["winner"] == "Democratic", ind=results["winner"] == "Independent",
                        rep=results["winner"] == "Republican")
         .groupby(["model", "draw"])[["dem", "ind", "rep"]].sum())
seats["dem"] += DEM_NOT_UP
seats["rep"] += REP_NOT_UP
assert (seats.sum(axis=1) == 100).all(), "Seats do not add up to 100"
chamber = seats.groupby("model").agg(
    dem_median=("dem", "median"), ind_median=("ind", "median"), rep_median=("rep", "median"),
    rep_q05=("rep", lambda s: s.quantile(0.05)), rep_q95=("rep", lambda s: s.quantile(0.95)))
chamber["p_rep_control"] = (seats["rep"] >= 50).groupby("model").mean()        # VP breaks 50-50 ties
chamber["p_dem_51_without_ind"] = (seats["dem"] >= 51).groupby("model").mean()  # no independent needed
chamber["national_sd"] = args.national_sd
control = chamber["p_rep_control"]

#### Checks and save ####
assert forecast["share"].between(0, 1).all()
assert np.isclose(forecast.groupby(["draw", "state", "model"])["share"].sum(), 1).all()

config.RESULTS.mkdir(parents=True, exist_ok=True)
config.write_table(forecast, config.FORECAST_FILE)
comparison.round(3).to_csv(config.RESULTS / "state_comparison.csv", index=False)
polls_df.to_csv(config.RESULTS / "poll_averages.csv", index=False)
chamber.round(3).to_csv(config.RESULTS / "chamber_summary.csv")
if len(gap_table):
    gap_table.to_csv(config.RESULTS / "calibration_gaps.csv", index=False)

print(f"\nSaved {len(forecast):,} rows to {config.FORECAST_FILE}")
print(f"Saved the comparison table to {config.RESULTS / 'state_comparison.csv'}")
print(f"Polls matched for {len(polls_df)} of {len(nominees)} states")
print(f"\nSenate by model (national error SD = {args.national_sd}). Republicans control with 50+ seats:")
print(chamber.round(2).to_string())
print("\nFirst rows of the comparison table:")
print(comparison.round(2).head(8).to_string(index=False))
