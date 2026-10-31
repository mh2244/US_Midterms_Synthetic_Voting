#### Preamble ####
# Purpose: Makes the forecast figure for the website: one quantile dotplot with one row per
#          state, pooling the four models (each model's 1,000 draws count equally). Each dot is
#          5% of the simulated outcomes (20 dots per state). The per-model numbers are in the
#          win-probability table from script 15.
# Author: (group)
# Date: 22 October 2026
# License: MIT
# Pre-requisites: Run scripts/11-build_forecast.py first. Needs: pip install matplotlib
# Any other information needed? Run from the project root. Output: figures/run_<RUN>/forecast_dotplot.png
#
# WHY a quantile dotplot: people read probabilities from dot counts more accurately than from
# density curves (Kay et al. 2016). Counting the dots on each side of 50% gives the chance of
# each outcome without a decimal that suggests false precision (Gelman et al. 2020).
# Assignment rules: blue for Democrats, red for Republicans, no boxplots.
# LIMIT: a win is coloured by the party that holds the dot's side of 50%. Where the left
# candidate is an independent (Nebraska, Idaho, South Dakota), the dots are grey instead
# of blue and the state label says so.


#### Workspace setup ####
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

BLUE, RED, GREY = (config.PARTY_COLOURS[k] for k in ("Democratic", "Republican", "Independent"))
N_DOTS = 20
OUT = config.FIGURES
OUT.mkdir(parents=True, exist_ok=True)

forecast = config.read_table(config.FORECAST_FILE)
print(f"Figure from run {config.RUN}")
nominees = pd.read_csv(config.NOMINEES_FILE).set_index("state")
republican = forecast[forecast["party"] == "Republican"]    # Republican vote share
models = sorted(republican["model"].unique())

# WHY pooled: with the inverse-variance combination the models agree closely, and four panels
# side by side were too small to read on the page. Pooling gives one readable panel.
order = republican.groupby("state")["share"].mean().sort_values().index.tolist()   # lower Republican shares first; higher shares appear at the top
q = (np.arange(N_DOTS) + 0.5) / N_DOTS
win = (republican.assign(rep_win=republican["winner"] == "Republican")
       .groupby(["state", "model"])["rep_win"].mean().groupby("state").mean())  # models count equally

config.use_site_style()            # shared graph theme: scripts/site.mplstyle
fig, ax = plt.subplots(figsize=(8, 0.28 * len(order) + 1.4))
for row, state in enumerate(order):
    draws = republican[republican["state"] == state]["share"].to_numpy()
    dots = np.quantile(draws, q)
    left_color = BLUE if nominees.loc[state, "left_party"] == "Democratic" else GREY
    ax.scatter(dots * 100, np.full(N_DOTS, row), c=[RED if d > 0.5 else left_color for d in dots],
               s=26, linewidths=0, zorder=3)
    pct = win[state] * 100
    text = "<1%" if pct < 0.5 else ">99%" if pct > 99.5 else f"{pct:.0f}%"
    ax.text(101, row, text, va="center", ha="left", fontsize=9, color=config.INK)
ax.axvline(50, color=config.INK, lw=1.2, ls=(0, (3, 2)), zorder=2)   # 50% line, as in every figure
ax.set_xlim(0, 100)
ax.set_ylim(-0.7, len(order) - 0.3)
ax.set_yticks(range(len(order)),
              [config.STATE_NAME.get(st, st) + ("*" if nominees.loc[st, "left_party"] == "Independent" else "")
               for st in order], fontsize=9)
ax.tick_params(axis="y", length=0)
ax.text(101, len(order) - 0.2, "Republican winning\nprobability", va="bottom", ha="left", fontsize=8,
        color=config.MUTED)
ax.set_xlabel("Republican candidate's share of the two-party vote (%)", fontsize=9)
ax.spines[["top", "right", "left"]].set_visible(False)
fig.tight_layout(rect=(0, 0, 0.93, 1))
fig.savefig(OUT / "forecast_dotplot.png", dpi=200)
print(f"Saved {OUT / 'forecast_dotplot.png'}")
