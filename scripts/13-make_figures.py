#### Preamble ####
# Purpose: Makes the forecast figure for the website: one quantile dotplot per model, with one
#          row per state. Each dot is 5% of the simulated outcomes (20 dots per state).
# Author: (group)
# Date: 22 October 2026
# License: MIT
# Pre-requisites: Run scripts/11-build_forecast.py first. Needs: pip install matplotlib
# Any other information needed? Run from the project root. Output: docs/figures/forecast_dotplot.png
#
# WHY a quantile dotplot: people read probabilities from dot counts more accurately than from
# density curves (Kay et al. 2016). Counting the dots on each side of 50% gives the chance of
# each outcome without a decimal that suggests false precision (Gelman et al. 2020).
# Assignment rules: blue for Democrats, red for Republicans, no boxplots.
# LIMIT: a win is coloured by the party that holds the dot's side of 50%. Where the left
# candidate is an independent (Nebraska, Idaho, South Dakota), the dots are blue-grey instead
# of blue and the state label says so.


#### Workspace setup ####
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

BLUE, RED, GREY_BLUE = "#2166ac", "#b2182b", "#5b7f95"
N_DOTS = 20
OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

forecast = config.read_table(config.FORECAST_FILE)
nominees = pd.read_csv(config.NOMINEES_FILE).set_index("state")
left = forecast[forecast["party"] != "Republican"]          # the left candidate's share
models = sorted(left["model"].unique())

# Order states from most Republican to most Democratic (average over models)
order = left.groupby("state")["share"].mean().sort_values().index.tolist()

fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 0.34 * len(order) + 1.6),
                         sharey=True, squeeze=False)
q = (np.arange(N_DOTS) + 0.5) / N_DOTS

for ax, model in zip(axes[0], models):
    for row, state in enumerate(order):
        draws = left[(left["model"] == model) & (left["state"] == state)]["share"].to_numpy()
        if len(draws) == 0:
            continue
        dots = np.quantile(draws, q)
        left_color = BLUE if nominees.loc[state, "left_party"] == "Democratic" else GREY_BLUE
        colors = [left_color if d > 0.5 else RED for d in dots]
        ax.scatter(dots * 100, np.full(N_DOTS, row), c=colors, s=22, linewidths=0)
    ax.axvline(50, color="#444444", lw=1)
    ax.set_xlim(0, 100)
    ax.set_title(model.replace("_", " "), fontsize=11, loc="left")
    ax.set_xlabel("Left candidate's share of the two-party vote (%)")
    ax.spines[["top", "right"]].set_visible(False)

labels = [s + ("*" if nominees.loc[s, "left_party"] == "Independent" else "") for s in order]
axes[0][0].set_yticks(range(len(order)))
axes[0][0].set_yticklabels(labels, fontsize=8)
fig.suptitle("Simulated Senate outcomes by state (each dot = 5% of draws; synthetic respondents)",
             x=0.01, ha="left", fontsize=11)
fig.text(0.01, 0.005, "* independent candidate on the left side. Blue = Democratic, red = Republican. "
         "Test run with few personas per state: not a forecast.", fontsize=8, ha="left")
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
fig.savefig(OUT / "forecast_dotplot.png", dpi=200)
print(f"Saved {OUT / 'forecast_dotplot.png'}")
