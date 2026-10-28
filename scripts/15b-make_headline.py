#### Preamble ####
# Purpose: Writes the headline at the top of the website and the seat bar under it, from the
#          forecast draws of this run, with the four models pooled (each model's draws count
#          equally, as in the dot plot from script 13).
# Author: Kexin (Coco) Liu
# Date: 26 October 2026
# License: MIT
# Pre-requisites: Run scripts/11-build_forecast.py (the default --combine ivw last).
# Any other information needed? Run from the project root:  python scripts/15b-make_headline.py
# Outputs: figures/run_<RUN>/seat_bar.png, data/03-results/run_<RUN>/headline.md and headline.json
#   (script 17 copies them to the latest folders that index.qmd reads).
#
# HOW the headline is chosen:
#   Democrats are favoured if they win 51 or more seats WITHOUT any independent in more than half
#   of the simulations; Republicans are favoured if they control the Senate (50 or more seats, with
#   Vice President Vance breaking ties) in more than half; otherwise the race is a toss-up.
# WHY Democrats need 51 on their own: Osborn (Nebraska) has said he will not caucus with either
#   party, so a 50-seat Democratic caucus that needs him is not counted as Democratic control. The
#   simulations in which Democrats hold 50 and an independent wins are in neither group.
# The seat numbers are pooled medians: Republican and independent seats are the medians over all
#   draws and models, and Democrats get the rest, so the bar always adds up to 100.

#### Workspace setup ####
import json
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

DEM_NOT_UP, REP_NOT_UP = 34, 31   # seats not up in 2026; same numbers as script 11
COL = config.PARTY_COLOURS
WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

#### Seats in every draw ####
forecast = config.read_table(config.FORECAST_FILE)
results = forecast[["model", "draw", "state", "winner"]].drop_duplicates()
seats = (results.assign(dem=results["winner"] == "Democratic", ind=results["winner"] == "Independent",
                        rep=results["winner"] == "Republican")
         .groupby(["model", "draw"])[["dem", "ind", "rep"]].sum())
seats["dem"] += DEM_NOT_UP
seats["rep"] += REP_NOT_UP
assert (seats.sum(axis=1) == 100).all(), "Seats do not add up to 100"
n_models = seats.index.get_level_values("model").nunique()
assert seats.groupby("model").size().nunique() == 1, "Models have different numbers of draws"

# Pooled over models: every model has the same number of draws, so each counts equally
p_dem = float((seats["dem"] >= 51).mean())       # Democratic control without independents
p_rep = float((seats["rep"] >= 50).mean())       # Republican control (VP breaks 50-50 ties)
rep = int(round(seats["rep"].median()))
ind = int(round(seats["ind"].median()))
dem = 100 - rep - ind
q05, q95 = (int(round(seats["rep"].quantile(x))) for x in (0.05, 0.95))

#### Headline text ####
tail = f", with {ind} independent" + ("s" if ind > 1 else "") if ind else ""
if p_dem > 0.5:
    head = f"Democrats are favoured to win the Senate, {dem} seats to {rep}{tail}"
elif p_rep > 0.5:
    head = f"Republicans are favoured to keep the Senate, {rep} seats to {dem}{tail}"
else:
    head = f"The Senate is a toss-up: {dem} Democratic seats to {rep} Republican{tail}"
day = "with no news" if config.RUN == "nonews" else \
    f"{date.fromisoformat(config.RUN).day} {date.fromisoformat(config.RUN):%B %Y}"
sub = (f"Democrats win 51 or more seats without independents in {p_dem * 100:.0f} per cent of "
       f"simulations, and Republicans keep control in {p_rep * 100:.0f} per cent. Republicans win "
       f"between {q05} and {q95} seats in 90 per cent of simulations. Synthetic voters from "
       f"{WORDS.get(n_models, n_models)} language models, combined with polls; run of {day}.")

#### Seat bar ####
config.use_site_style()            # shared graph theme: scripts/site.mplstyle
fig, ax = plt.subplots(figsize=(8, 1.5))
left = 0
for n, party in [(dem, "Democratic"), (ind, "Independent"), (rep, "Republican")]:
    if n:
        ax.barh(0, n, left=left, height=0.6, color=COL[party], edgecolor="white", linewidth=2)
    left += n
ax.vlines(50, -0.4, 0.4, color=config.INK, lw=1.2, ls=(0, (3, 2)), zorder=3)
ax.text(50, 0.44, "50 seats and the vice president", ha="center", va="bottom", fontsize=8, color=config.MUTED)
ax.text(0, -0.42, f"Democrats {dem}", ha="left", va="top", fontsize=12, fontweight="bold", color=COL["Democratic"])
ax.text(100, -0.42, f"Republicans {rep}", ha="right", va="top", fontsize=12, fontweight="bold",
        color=COL["Republican"])
if ind:
    ax.text(dem + ind / 2, -0.42, f"Independent {ind}", ha="center", va="top", fontsize=10,
            color=COL["Independent"])
ax.set_xlim(0, 100)
ax.set_ylim(-0.9, 0.75)
ax.axis("off")
fig.tight_layout(pad=0.3)
config.FIGURES.mkdir(parents=True, exist_ok=True)
fig.savefig(config.FIGURES / "seat_bar.png", dpi=200)
plt.close(fig)

#### Save ####
md = (f"## {head}\n\n{sub}\n\n"
      "![Median seats in the next Senate, all simulations and models pooled.](figures/latest/seat_bar.png)"
      '{fig-alt="Horizontal bar of 100 Senate seats split into Democratic, independent and Republican '
      f'seats ({dem}, {ind} and {rep}), with a dashed line at 50."}}\n')
config.RUN_RESULTS.mkdir(parents=True, exist_ok=True)
(config.RUN_RESULTS / "headline.md").write_text(md)
(config.RUN_RESULTS / "headline.json").write_text(json.dumps(
    {"run": config.RUN, "dem": dem, "ind": ind, "rep": rep, "p_dem_51_without_ind": round(p_dem, 3),
     "p_rep_control": round(p_rep, 3), "rep_q05": q05, "rep_q95": q95, "n_models": n_models}, indent=2) + "\n")

assert dem + ind + rep == 100 and min(dem, ind, rep) >= 0
assert 0 <= p_dem <= 1 and 0 <= p_rep <= 1 and p_dem + p_rep <= 1 + 1e-9, "Control chances do not add up"
print(head)
print(sub)
print(f"Saved {config.FIGURES / 'seat_bar.png'} and {config.RUN_RESULTS / 'headline.md'}")
