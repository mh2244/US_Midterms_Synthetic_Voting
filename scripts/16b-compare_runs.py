#### Preamble ####
# Purpose: Compares the forecast runs with each other and with simpler methods, for the section
#          "What the synthetic voters add" on the website. Three tables:
#            1. runs.md     the headline numbers for each news day and the no-news baseline
#            2. news.md     the synthetic voters' share before any poll correction, by model and
#                           run (also drawn in figures/run_<RUN>/news_effect.png),
#                           run, and how far each news day moved it from the no-news baseline
#            3. methods.md  for the latest news run: chance of Republican control and median seats
#                           under three methods (synthetic only, inverse-variance, and 2024 +
#                           uniform swing with no language model)
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
#### Workspace setup ####
from pathlib import Path

import numpy as np
import pandas as pd

import config

NEWS_DAYS = ["2026-10-27", "2026-10-29", "2026-10-31"]
OUT = config.RESULTS / "comparison"
DEM_NOT_UP, REP_NOT_UP = 34, 31          # as in script 11
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
    metadata = pd.read_csv(config.RESULTS / f"run_{run}" / "chamber_summary_ivw.csv")
    raw_method = ("bias_correction" in metadata and
                  metadata["bias_correction"].astype(str).str.lower().eq("false").all())
    rows.append({"run": run, "combination": "raw synthetic" if raw_method else "mean-bias corrected", **s})
by_run = pd.DataFrame(rows)
by_run.round(4).to_csv(OUT / "runs.csv", index=False)
md = ["| Day | Probability of Democratic caucus control (%) | Probability of Republican control (%) | Republican seats, median (90% interval) |",
      "|---|---:|---:|---:|"]
for r in by_run[by_run["run"] != "nonews"].itertuples():   # dated runs only; the no-news run stays in runs.csv
    md.append(f"| {label(r.run)} | {pct(r.p_dem)} | {pct(r.p_rep)} | {r.rep_median} ({r.rep_q05} to {r.rep_q95}) |")
(OUT / "runs.md").write_text("\n".join(md) + "\n")

#### 2. Synthetic share before any poll correction, by model and run ####
# Montana is left out of the news comparison on every day: its left candidate changed from the
# Democrat Alani Bankhead (27 and 29 October and the no-news run) to the independent Seth Bodnar
# (31 October), so its change from the no-news run is not an effect of the news.
NEWS_EXCLUDE = ["MT"]
share = {}
for run in runs:
    comp = pd.read_csv(config.RESULTS / f"run_{run}" / "state_comparison_uncalibrated.csv")
    comp = comp[~comp["state"].isin(NEWS_EXCLUDE)]
    cols = [c for c in comp.columns if c.startswith("model_")]
    share[run] = comp[cols].mean().rename(lambda c: c.removeprefix("model_"))   # plain mean over the 34 states
news = pd.DataFrame(share)
models = [m for m in config.MODEL_LABELS if m in news.index] + sorted(set(news.index) - set(config.MODEL_LABELS))
news = news.loc[models]
news.loc["all"] = news.mean()                                                    # each model counts equally
news.round(4).to_csv(OUT / "news.csv")
republican_news = 1 - news  # Display only; news.csv retains its original left-share schema.
head = ["Model"] + [label(r) for r in runs] + ([f"{label(r)} minus no news" for r in news_runs] if "nonews" in runs else [])
md = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
for m, r in republican_news.iterrows():
    name = "**All models**" if m == "all" else config.MODEL_LABELS.get(m, m)
    cells = [pct(r[run]) for run in runs]
    if "nonews" in runs:
        cells += [signed(r[run] - r["nonews"]) for run in news_runs]
    md.append("| " + " | ".join([name] + cells) + " |")
(OUT / "news.md").write_text("\n".join(md) + "\n")

#### 2b. Figure: the synthetic share in each setting, and each state's change from no news ####
# Left: mean over the 34 states (Montana left out, see section 2) for each model (config.MODEL_COLOURS, labelled at the right) and all models (black),
# in the order no news, 27, 29, 31 October. Right (when the no-news run exists): each state's change
# from no news, all models pooled, one dot per state. Saved as figures/run_<RUN>/news_effect.png.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

config.use_site_style()
order = (["nonews"] if "nonews" in runs else []) + news_runs
xlab = ["No news" if r == "nonews" else label(r) for r in order]
per_state = {}
for run in order:
    comp = pd.read_csv(config.RESULTS / f"run_{run}" / "state_comparison_uncalibrated.csv").set_index("state")
    comp = comp.drop(index=NEWS_EXCLUDE, errors="ignore")
    per_state[run] = 1 - comp[[c for c in comp.columns if c.startswith("model_")]].mean(axis=1)   # models pooled
has_base = "nonews" in runs and len(news_runs) > 0
fig, axes = plt.subplots(1, 2 if has_base else 1, figsize=(9, 3.6) if has_base else (5.5, 3.6),
                         gridspec_kw={"width_ratios": [1.15, 1]} if has_base else None)
ax = axes[0] if has_base else axes
x = np.arange(len(order))
greys = ["#9a9a9a", "#b8b8b8", "#7c7c7c", "#cfcfcf"]
markers = ["o", "s", "^", "D"]
labels = []                                 # (y at the last setting, text, colour)
for i, m in enumerate([m for m in news.index if m != "all"]):
    y = republican_news.loc[m, order].to_numpy() * 100
    colour = config.MODEL_COLOURS.get(m, greys[i % 4])
    ax.plot(x, y, color=colour, marker=markers[i % 4], ms=4, lw=1.4, zorder=3)   # above the pooled line
    labels.append([y[-1], config.MODEL_LABELS.get(m, m), colour])
y = republican_news.loc["all", order].to_numpy() * 100
ax.plot(x, y, color=config.INK, marker="o", ms=7, lw=3, zorder=2)
labels.append([y[-1], "All models", None])
labels.sort(key=lambda t: t[0])             # spread the labels so that none overlap (at least 0.55 points apart)
for k in range(1, len(labels)):
    labels[k][0] = max(labels[k][0], labels[k - 1][0] + 0.55)
for y_lab, text, colour in labels:          # colour None = the pooled line, in black and bold
    ax.text(x[-1] + 0.12, y_lab, text, va="center", fontsize=8.5 if colour is None else 8,
            fontweight="bold" if colour is None else "normal", color=config.INK if colour is None else colour)
mid = republican_news.loc[:, order].to_numpy().mean() * 100
ax.set_ylim(mid - 5, mid + 5)               # a 10-point window, so small moves look small
ax.set_xlim(-0.2, len(order) - 1 + 1.4)
ax.set_xticks(x, xlab)
ax.set_ylabel("Synthetic Republican share (%)")
ax.set_title("Average Republican share, by model")
if has_base:
    ax2 = axes[1]
    days = news_runs
    for j, run in enumerate(days):
        d = (per_state[run] - per_state["nonews"]) * 100
        jitter = np.random.default_rng(j).uniform(-0.12, 0.12, len(d))   # own generator: leaves rng for section 3
        ax2.scatter(np.full(len(d), j) + jitter, d, s=12, color=config.INK, alpha=0.55, lw=0)
        ax2.plot([j - 0.25, j + 0.25], [d.median()] * 2, color=config.INK, lw=2)
    ax2.axhline(0, color=config.INK, lw=1.2, ls=(0, (3, 2)))
    lim = max(3, float(np.ceil(max(abs((per_state[r] - per_state["nonews"]) * 100).max() for r in days))))
    ax2.set_ylim(-lim, lim)
    ax2.set_xlim(-0.6, len(days) - 0.4)
    ax2.set_xticks(range(len(days)), [label(r) for r in days])
    ax2.set_ylabel("Change from no news (points)")
    ax2.set_title("Change from no news, by state")
fig.tight_layout()
config.FIGURES.mkdir(parents=True, exist_ok=True)
fig.savefig(config.FIGURES / "news_effect.png")
plt.close(fig)

#### 3. Methods, for the latest news run ####
res = config.RESULTS / f"run_{latest}"
methods = []
for tag, name in [("uncalibrated", "Synthetic voters only"), ("ivw", "Synthetic + polling information")]:
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
methods.append({"method": "Polls + 2024 swing (no LLM)",
                **summary(seats_from(pd.concat(sim, ignore_index=True)))})

meth = pd.DataFrame(methods)
meth.round(4).to_csv(OUT / "methods.csv", index=False)
# This paragraph follows the latest run, rather than leaving old figures in the Discussion.
synthetic, combined, baseline = methods
polled_weights = w.loc[w["source"] == "poll", "w_synthetic"].mean()
unpolled_weights = w.loc[w["source"] != "poll", "w_synthetic"].mean()
summary_text = (
    f"For {label(latest)}, adding polling and historical information lowers the Republican "
    f"control probability from {pct(synthetic['p_rep'])}% in the synthetic-only forecast to "
    f"{pct(combined['p_rep'])}%. The no-LLM benchmark gives {pct(baseline['p_rep'])}%, so including "
    "synthetic responses moves the combined forecast toward greater Republican support. "
    f"Synthetic estimates receive an average weight of {polled_weights * 100:.0f}% in polled "
    f"states and {unpolled_weights * 100:.0f}% in unpolled states. The remaining weight goes "
    "to state polling averages or, where unavailable, the 2024 CES vote share plus uniform swing. "
    "The combined result is therefore driven mainly by these outside estimates."
)
(OUT / "method_summary.md").write_text(summary_text + "\n")

md = ["| Method | Probability of Republican control (%) | Republican seats, median (90% interval) |",
      "|---|---:|---:|"]
for r in meth.itertuples():
    md.append(f"| {r.method} | {pct(r.p_rep)} | {r.rep_median} ({r.rep_q05} to {r.rep_q95}) |")
(OUT / "methods.md").write_text("\n".join(md) + "\n")

(OUT / "latest_run.txt").write_text(label(latest) + "\n")

#### Checks ####
assert by_run[["p_dem", "p_rep"]].stack().between(0, 1).all()
assert (by_run["p_dem"] + by_run["p_rep"] <= 1 + 1e-9).all(), "Control chances add up to more than 1"
assert news[runs].stack().between(0, 1).all(), "A synthetic share is outside 0 to 1"
assert len(meth) == 3 and meth[["p_dem", "p_rep"]].stack().between(0, 1).all()

print((OUT / "runs.md").read_text())
print((OUT / "news.md").read_text())
print((OUT / "methods.md").read_text())
