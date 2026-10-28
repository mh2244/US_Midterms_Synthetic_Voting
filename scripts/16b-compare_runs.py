#### Preamble ####
# Purpose: Compares the forecast runs with each other and with simpler methods, for the section
#          "What the synthetic voters add" on the website. Three tables:
#            1. runs.md     the headline numbers for each news day and the no-news baseline
#            2. news.md     the synthetic voters' share before any poll correction, by model and
#                           run, and how far each news day moved it from the no-news baseline
#            3. methods.md  for the latest news run: chance of Republican control and median seats
#                           under four methods (synthetic only, shrinkage, inverse-variance, and
#                           2024 + uniform swing with no language model), plus the leave-one-out
#                           error of each method against the polls
# Author: Kexin (Coco) Liu
# Date: 27 October 2026
# License: MIT
# Pre-requisites: Run scripts 11 (all three --combine versions) for each run to compare.
# Any other information needed? Run from the project root:  python scripts/16b-compare_runs.py
#   It reads every run in data/03-results/ that is a planned news day or "nonews" and writes to
#   data/03-results/comparison/. Runs that do not exist yet are left out, so it can run any day.
#   The 26 October run was a test of the full pipeline and is not compared.
#
# HOW the numbers are pooled: as in scripts 13 and 15b, each model's 1,000 draws count equally.
# HOW the "2024 + uniform swing" method is simulated: each state's share is drawn around its
#   outside estimate t from script 11 (the poll average in polled states, 2024 + swing elsewhere)
#   with that estimate's standard deviation, plus the same national error as script 11 (0.03).
#   It is the inverse-variance forecast with the synthetic voters' weight set to zero.
# HOW the leave-one-out error is computed: each polled state's poll is hidden in turn, the bias,
#   swing and variances are estimated from the other polled states, and the hidden poll average is
#   predicted. The target is a poll average, not a result, so this measures agreement with polls.

#### Workspace setup ####
from pathlib import Path

import numpy as np
import pandas as pd

import config

NEWS_DAYS = ["2026-10-27", "2026-10-29", "2026-10-31"]
OUT = config.RESULTS / "comparison"
DEM_NOT_UP, REP_NOT_UP = 34, 31          # as in script 11
FLOOR = 0.005 ** 2                         # as in script 11
rng = np.random.default_rng(config.SEED)


def label(run):
    if run == "nonews":
        return "No news (baseline)"
    d = pd.Timestamp(run)
    return f"{d.day} {d:%B}"


def read_forecast(path):
    path = Path(path)
    return pd.read_parquet(path) if path.exists() else pd.read_csv(path.with_suffix(".csv"))


def seats_from(results):
    """Seats per (model, draw) from rows of state winners."""
    s = (results.assign(dem=results["winner"] == "Democratic", ind=results["winner"] == "Independent",
                        rep=results["winner"] == "Republican")
         .groupby(["model", "draw"])[["dem", "ind", "rep"]].sum())
    s["dem"] += DEM_NOT_UP
    s["rep"] += REP_NOT_UP
    assert (s.sum(axis=1) == 100).all(), "Seats do not add up to 100"
    return s


def summary(seats):
    return {"p_dem": (seats["dem"] >= 51).mean(), "p_rep": (seats["rep"] >= 50).mean(),
            "rep_median": int(round(seats["rep"].median())),
            "rep_q05": int(round(seats["rep"].quantile(0.05))), "rep_q95": int(round(seats["rep"].quantile(0.95)))}


def pct(x):
    return f"{x * 100:.0f}"


def signed(x):
    v = round(x * 100, 1)
    return "0.0" if v == 0 else f"+{v:.1f}" if v > 0 else f"−{-v:.1f}"


#### Which runs exist ####
runs = [r for r in NEWS_DAYS + ["nonews"] if (config.RESULTS / f"run_{r}" / "chamber_summary_ivw.csv").exists()]
news_runs = [r for r in runs if r != "nonews"]
assert news_runs, "No news-day run with an inverse-variance forecast yet"
latest = news_runs[-1]
print(f"Comparing runs: {', '.join(runs)}; latest news run: {latest}")
OUT.mkdir(parents=True, exist_ok=True)

#### 1. Headline numbers by run ####
rows = []
for run in runs:
    f = read_forecast(config.RESULTS / f"run_{run}" / "forecast_ivw.parquet")
    s = summary(seats_from(f[["model", "draw", "state", "winner"]].drop_duplicates()))
    rows.append({"run": run, **s})
by_run = pd.DataFrame(rows)
by_run.round(4).to_csv(OUT / "runs.csv", index=False)
md = ["| Run | Democrats win 51+ seats (%) | Republicans keep control (%) | Republican seats, median (90% interval) |",
      "|---|---:|---:|---:|"]
for r in by_run.itertuples():
    md.append(f"| {label(r.run)} | {pct(r.p_dem)} | {pct(r.p_rep)} | {r.rep_median} ({r.rep_q05} to {r.rep_q95}) |")
(OUT / "runs.md").write_text("\n".join(md) + "\n")

#### 2. Synthetic share before any poll correction, by model and run ####
share = {}
for run in runs:
    comp = pd.read_csv(config.RESULTS / f"run_{run}" / "state_comparison_uncalibrated.csv")
    cols = [c for c in comp.columns if c.startswith("model_")]
    share[run] = comp[cols].mean().rename(lambda c: c.removeprefix("model_"))   # plain mean over the 35 states
news = pd.DataFrame(share)
models = [m for m in config.MODEL_LABELS if m in news.index] + sorted(set(news.index) - set(config.MODEL_LABELS))
news = news.loc[models]
news.loc["all"] = news.mean()                                                    # each model counts equally
news.round(4).to_csv(OUT / "news.csv")
head = ["Model"] + [label(r) for r in runs] + ([f"{label(r)} minus no news" for r in news_runs] if "nonews" in runs else [])
md = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
for m, r in news.iterrows():
    name = "**All models**" if m == "all" else config.MODEL_LABELS.get(m, m)
    cells = [pct(r[run]) for run in runs]
    if "nonews" in runs:
        cells += [signed(r[run] - r["nonews"]) for run in news_runs]
    md.append("| " + " | ".join([name] + cells) + " |")
(OUT / "news.md").write_text("\n".join(md) + "\n")

#### 3. Methods, for the latest news run ####
res = config.RESULTS / f"run_{latest}"
methods = []
for tag, name in [("uncalibrated", "Synthetic voters only"), ("calibrated", "Shrinkage toward polls"),
                  ("ivw", "Inverse-variance combination (headline)")]:
    f = read_forecast(res / f"forecast_{tag}.parquet")
    methods.append({"method": name, **summary(seats_from(f[["model", "draw", "state", "winner"]].drop_duplicates()))})

# 2024 + uniform swing, no language model: draw around t with sd_t (same for every model)
w = pd.read_csv(res / "ivw_weights.csv")
base = w.drop_duplicates("state").set_index("state")
nominees = pd.read_csv(config.NOMINEES_FILE).set_index("state")
draws = len(read_forecast(res / "forecast_ivw.parquet")["draw"].unique())
n = draws * w["model"].nunique()
shift = rng.normal(0, config.NATIONAL_SD, n)
sim = []
for state, r in base.iterrows():
    sd_t = np.sqrt(r["var_t"])
    s = np.clip(r["t"] + sd_t * rng.standard_normal(n) + shift, 0, 1)
    left = nominees.loc[state, "left_party"]
    sim.append(pd.DataFrame({"model": "baseline", "draw": np.arange(n), "state": state,
                             "winner": np.where(s > 0.5, left, "Republican")}))
methods.append({"method": "2024 + uniform swing (no language model)",
                **summary(seats_from(pd.concat(sim, ignore_index=True)))})

# Leave-one-out error against the polls, pooled over models
poll = pd.read_csv(res / "state_comparison_ivw.csv")[["state", "poll_share", "n_polls", "pres2024_dem_share"]]
loo = []
for model, g in w.merge(poll, on="state").groupby("model"):
    P = g[g["source"] == "poll"].copy()
    P["var_poll"] = P["var_t"]
    for s in P["state"]:
        tr, te = P[P["state"] != s], P[P["state"] == s].iloc[0]
        gap = (tr["model_share"] - tr["poll_share"]).mean()
        swing = (tr["poll_share"] - tr["pres2024_dem_share"]).mean()
        vs = max(FLOOR, (tr["model_share"] - gap - tr["poll_share"]).var() - tr["var_boot"].mean() - tr["var_poll"].mean())
        vb = max(FLOOR, (tr["poll_share"] - tr["pres2024_dem_share"] - swing).var() - tr["var_poll"].mean())
        m_star, b = te["model_share"] - gap, te["pres2024_dem_share"] + swing
        vm = te["var_boot"] + vs
        wt = vm / (vm + vb)
        # shrinkage with no poll for this state: unpolled states get 2024 + swing, as in script 11
        loo.append({"model": model, "state": s, "truth": te["poll_share"], "synthetic": te["model_share"],
                    "baseline": b, "shrink": b, "ivw": wt * b + (1 - wt) * m_star})
loo = pd.DataFrame(loo)
err = {k: (loo[k] - loo["truth"]).abs().mean() for k in ["synthetic", "shrink", "ivw", "baseline"]}
loo.round(4).to_csv(OUT / "leave_one_out.csv", index=False)
for m, k in zip(methods, ["synthetic", "shrink", "ivw", "baseline"]):
    m["loo_error"] = err[k]
meth = pd.DataFrame(methods)
meth.round(4).to_csv(OUT / "methods.csv", index=False)
md = ["| Method | Republicans keep control (%) | Republican seats, median (90% interval) | Error against hidden polls (points) |",
      "|---|---:|---:|---:|"]
for r in meth.itertuples():
    md.append(f"| {r.method} | {pct(r.p_rep)} | {r.rep_median} ({r.rep_q05} to {r.rep_q95}) | {r.loo_error * 100:.1f} |")
(OUT / "methods.md").write_text("\n".join(md) + "\n")
(OUT / "latest_run.txt").write_text(label(latest) + "\n")

#### Checks ####
assert by_run[["p_dem", "p_rep"]].stack().between(0, 1).all()
assert (by_run["p_dem"] + by_run["p_rep"] <= 1 + 1e-9).all(), "Control chances add up to more than 1"
assert news[runs].stack().between(0, 1).all(), "A synthetic share is outside 0 to 1"
assert meth["loo_error"].notna().all() and len(loo) > 0

print((OUT / "runs.md").read_text())
print((OUT / "news.md").read_text())
print((OUT / "methods.md").read_text())
