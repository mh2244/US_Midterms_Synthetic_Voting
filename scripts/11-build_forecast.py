#### Preamble ####
# Purpose: Turns persona-level probabilities into a distribution of outcomes for each state and
#          model, saves it in the same layout as the simulated forecast, and builds a table
#          that sets the model's share beside the poll average and the 2024 presidential vote.
# Author: (group)
# Date: 22 October 2026
# License: MIT
# Pre-requisites: Run scripts/10-score_responses.py first. Needs pyarrow for the parquet file.
# Any other information needed? Run from the project root.
#   python scripts/11-build_forecast.py                   <- headline: inverse-variance combination
#   python scripts/11-build_forecast.py --combine none    <- comparison: synthetic voters only
#   Each run also saves labelled copies (forecast_ivw / _uncalibrated), and scripts
#   13 and 15 draw whichever ran last, so run the headline (ivw) last.
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
#   check that the code works, NOT a forecast. With 100 personas it is about 3 to 4 points
#   (26 October test run), because SSR probabilities are rarely exactly 0 or 1.
# HOW polls enter (default --combine ivw): see the comment above the inverse-variance block.
#   The bootstrap variance is one part of the synthetic variance; the rest is estimated from
#   how far the bias-corrected synthetic shares fall from the polls.
# A national error shared by all states is ON by default (config.NATIONAL_SD = 0.03, an
#   assumption; Gelman et al. 2020; Shirani-Mehr et al. 2018). Rerun with --national-sd 0, 0.02
#   and 0.04 for the sensitivity table, and --poll-sd 0.02 and 0.04 for the poll error.
# NOT included (say so in the paper): spread between models and repeats, uncertainty in the
#   estimated variances themselves, and correlation between synthetic and poll errors.
# HOW seats are counted: Nebraska, Idaho and South Dakota have an independent instead of a
#   Democrat, and Osborn (NE) has said he will not caucus with either party, so independent wins
#   are counted as their own category, not as Democratic seats. The headline is the chance that
#   Republicans keep control: with Vice President Vance breaking 50-50 ties, Republicans control
#   the Senate with 50 or more seats, whichever side the independents join.
# WHY polls are combined with, not replaced by, the model: the model is biased toward one side
# (Pew 2026; Heath & Alexander 2026). Polls are not the truth either, because poll errors are
# correlated across polls and offices (Shirani-Mehr et al. 2018).
# LIMIT: unpolled states lean on 2024 + a uniform swing, which ignores candidate effects there.


#### Workspace setup ####
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config

ap = argparse.ArgumentParser()
ap.add_argument("--design", default="B", choices=["B", "C"], help="which prompt design to forecast from")
ap.add_argument("--draws", type=int, default=config.N_DRAWS)
ap.add_argument("--combine", default="ivw", choices=["ivw", "none"],
                help="ivw: inverse-variance weights with polls or 2024 + swing (headline); "
                     "none: synthetic only")
ap.add_argument("--poll-sd", type=float, default=0.03,
                help="state poll error beyond sampling error, used by --combine ivw (assumption)")
ap.add_argument("--national-sd", type=float, default=config.NATIONAL_SD,
                help="SD of an error shared by all states (default in config.py; 0 = states independent)")
ap.add_argument("--poll-window-days", type=int, default=45)
ap.add_argument("--summary-only", type=Path, default=None,
                help="append the chamber summary to this csv and save nothing else (for the "
                     "sensitivity table, scripts/16c-sensitivity.py)")
args = ap.parse_args()
args.ivw = args.combine == "ivw"

# Seats not up in 2026, checked 2026-10-25: the Senate is 53 R - 47 D (the 47 include King and
# Sanders, independents who caucus with Democrats). 35 seats are up: 13 held by Democrats and 22
# by Republicans (including the FL and OH specials). 47 - 13 = 34 and 53 - 22 = 31.
DEM_NOT_UP, REP_NOT_UP = 34, 31

rng = np.random.default_rng(config.SEED)
nominees = pd.read_csv(config.NOMINEES_FILE).set_index("state")
personas = config.read_table(config.PERSONAS_FILE)

#### Persona-level probabilities ####
print(f"Run: {config.RUN}  (results in {config.RUN_RESULTS})")
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
        return pd.DataFrame(columns=["state", "poll_share", "n_polls", "n_sample"])
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
                     "n_polls": len(both), "n_sample": float(weight.sum())})
    return pd.DataFrame(rows, columns=["state", "poll_share", "n_polls", "n_sample"])


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

#### 2024 presidential baseline (CES, validated-vote weights) ####
# Used for the comparison table and for the uniform swing of unpolled states.
ces = pd.read_csv(config.CES_FILE, low_memory=False,
                  usecols=["state", "vvweight_post", "pres_vote_2024"])
ces = ces[ces["state"].isin(nominees.index) & ces["vvweight_post"].notna()
          & ces["pres_vote_2024"].isin(["Harris", "Trump"])]
w = ces.pivot_table(index="state", columns="pres_vote_2024", values="vvweight_post", aggfunc="sum")
pres = (w["Harris"] / (w["Harris"] + w["Trump"])).rename("pres2024_dem_share").reset_index()

# Inverse-variance combination (--combine ivw, the default and the headline)
#   Two noisy readings of each state's Democratic-side share are averaged, each weighted by
#   1 / its variance (the less noisy one counts more):
#     synthetic  m* = model share - mean gap to polls (the mean gap is the bias, removed first)
#                var_m = persona bootstrap variance + structural variance of the model
#     outside    t = poll average in polled states, 2024 presidential share + uniform swing in
#                unpolled states; var_t = poll variance, or the error of uniform swing
#     combined = w * t + (1 - w) * m*,  w = var_m / (var_m + var_t),  var = 1 / (1/var_m + 1/var_t)
#   The structural variance and the error of uniform swing are estimated from the polled states:
#     var(m* - poll)            = structural + bootstrap + poll   -> solve for structural
#     var(poll - (2024 + swing)) = swing error + poll              -> solve for swing error
#   Poll variance = sampling variance s(1-s)/n + args.poll_sd^2 (assumed error beyond sampling).
#   ASSUMES the errors are independent of each other; draws are normal around the combination.
weight_table = pd.DataFrame()
if args.ivw:
    pt = pd.DataFrame([{"model": p["model"], "state": p["state"], "model_share": p["model_share"],
                        "var_boot": float(np.var(p["raw_draws"]))} for p in point])
    pt = pt.merge(polls_df, on="state", how="left").merge(pres, on="state", how="left")
    pt["var_poll"] = (pt["poll_share"] * (1 - pt["poll_share"]) / pt["n_sample"]) + args.poll_sd ** 2
    floor = 0.005 ** 2                                   # keeps a variance from being estimated as 0
    for model, grp in pt.groupby("model"):
        polled = grp.dropna(subset=["poll_share"])
        if len(polled) < 8:
            print(f"--combine ivw skipped for {model}: only {len(polled)} polled states (need 8)")
            continue
        grp = grp.copy()
        mean_gap = float((polled["model_share"] - polled["poll_share"]).mean())
        swing = float((polled["poll_share"] - polled["pres2024_dem_share"]).mean())
        grp["m_star"] = grp["model_share"] - mean_gap
        resid_m = polled["model_share"] - mean_gap - polled["poll_share"]
        var_struct = max(floor, resid_m.var() - polled["var_boot"].mean() - polled["var_poll"].mean())
        resid_b = polled["poll_share"] - (polled["pres2024_dem_share"] + swing)
        var_swing = max(floor, resid_b.var() - polled["var_poll"].mean())
        grp["var_m"] = grp["var_boot"] + var_struct
        is_polled = grp["poll_share"].notna()
        grp["source"] = np.where(is_polled, "poll", "2024 + swing")
        grp["t"] = np.where(is_polled, grp["poll_share"], grp["pres2024_dem_share"] + swing)
        grp["var_t"] = np.where(is_polled, grp["var_poll"], var_swing)
        assert grp[["t", "var_t", "var_m"]].notna().all().all(), "Missing input to --ivw"
        w = grp["var_m"] / (grp["var_m"] + grp["var_t"])
        grp["w_synthetic"] = 1 - w
        grp["combined"] = w * grp["t"] + (1 - w) * grp["m_star"]
        grp["sd_combined"] = np.sqrt(1 / (1 / grp["var_m"] + 1 / grp["var_t"]))
        grp["mean_gap"], grp["swing"] = mean_gap, swing
        grp["sd_struct"], grp["sd_swing"] = np.sqrt(var_struct), np.sqrt(var_swing)
        lookup = grp.set_index("state")
        for p in point:
            if p["model"] == model:
                r = lookup.loc[p["state"]]
                p["base_draws"] = r["combined"] + r["sd_combined"] * rng.standard_normal(D)
        weight_table = pd.concat([weight_table, grp])
        print(f"{model}: bias (mean gap) {mean_gap:+.3f}; structural sd {np.sqrt(var_struct):.3f}; "
              f"swing {swing:+.3f}, swing error sd {np.sqrt(var_swing):.3f}; "
              f"synthetic weight polled {grp.loc[is_polled, 'w_synthetic'].mean():.2f}, "
              f"unpolled {grp.loc[~is_polled, 'w_synthetic'].mean():.2f}")

out = []
for p in point:
    base = p.get("base_draws", p["raw_draws"])
    share = np.clip(base + p["shift"], 0, 1)
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
chamber["poll_sd"] = args.poll_sd
chamber["combine"] = args.combine
control = chamber["p_rep_control"]

#### Checks and save ####
assert forecast["share"].between(0, 1).all()
assert np.isclose(forecast.groupby(["draw", "state", "model"])["share"].sum(), 1).all()

if args.summary_only:                  # sensitivity runs: leave the run's real outputs untouched
    out = args.summary_only
    out.parent.mkdir(parents=True, exist_ok=True)
    chamber.round(4).to_csv(out, mode="a", header=not out.exists())
    print(f"Appended the chamber summary to {out}")
    raise SystemExit(0)

config.RUN_RESULTS.mkdir(parents=True, exist_ok=True)
config.write_table(forecast, config.FORECAST_FILE)
comparison.round(3).to_csv(config.RUN_RESULTS / "state_comparison.csv", index=False)
polls_df.to_csv(config.RUN_RESULTS / "poll_averages.csv", index=False)
chamber.round(3).to_csv(config.RUN_RESULTS / "chamber_summary.csv")
# The files above are what scripts 13 and 15 read (the latest version run). Keep a labelled
# copy as well, so the synthetic-only and inverse-variance forecasts can be compared side by side.
tag = "ivw" if args.ivw else "uncalibrated"
config.write_table(forecast, config.RUN_RESULTS / f"forecast_{tag}.parquet")
comparison.round(3).to_csv(config.RUN_RESULTS / f"state_comparison_{tag}.csv", index=False)
chamber.round(3).to_csv(config.RUN_RESULTS / f"chamber_summary_{tag}.csv")
if len(weight_table):
    keep = ["model", "state", "source", "model_share", "mean_gap", "m_star", "t", "combined",
            "w_synthetic", "sd_combined", "var_boot", "var_m", "var_t", "sd_struct", "swing", "sd_swing"]
    weight_table[keep].round(4).to_csv(config.RUN_RESULTS / "ivw_weights.csv", index=False)

print(f"\nSaved {len(forecast):,} rows to {config.FORECAST_FILE}")
print(f"Saved the comparison table to {config.RUN_RESULTS / 'state_comparison.csv'}")
print(f"Polls matched for {len(polls_df)} of {len(nominees)} states")
print(f"\nSenate by model (national error SD = {args.national_sd}). Republicans control with 50+ seats:")
print(chamber.round(2).to_string())
print("\nFirst rows of the comparison table:")
print(comparison.round(2).head(8).to_string(index=False))
