#### Preamble ####
# Purpose: Summarizes forecast draws as state-level win probabilities and plots
#          stacked horizontal bars for the Democratic/independent and Republican sides.
# Author: Jingxuan Feng
# Date: 26 October 2026
# Contact: amelia.f.feng@outlook.com
# License: MIT
# Pre-requisites: Run 11-build_forecast.py; install pandas numpy matplotlib pyarrow.
# Run from the project root: python scripts/15-make_summary_bar.py
# Outputs: data/03-results/win_probabilities.csv and
#          docs/figures/forecast_win_probability_bar.png
# Each draw counts once, using its saved winner (including tie-break results).
# These are simulated win frequencies, not vote shares or calibrated election odds.

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import config

BLUE, RED, INDEPENDENT = "#2166ac", "#b2182b", "#5b7f95"


def summarize(forecast, nominees):
    """Count one saved winner per state/model/draw, checking both party rows."""
    required = {"state", "model", "draw", "party", "winner"}
    if forecast.empty or not required <= set(forecast.columns):
        raise ValueError("Forecast is empty or missing required columns")
    if nominees["state"].duplicated().any():
        raise ValueError("Nominees must contain one row per state")
    if forecast[list(required)].isna().any().any():
        raise ValueError("Forecast contains missing identifiers, parties or winners")
    keys = ["state", "model", "draw"]
    if forecast.duplicated(keys + ["party"]).any():
        raise ValueError("Duplicate party rows in a forecast draw")
    groups = forecast.groupby(keys)
    if not groups.size().eq(2).all() or not groups["winner"].nunique().eq(1).all():
        raise ValueError("Each draw must contain two party rows and one consistent winner")
    checked = forecast.merge(nominees[["state", "left_party"]], on="state", how="left", validate="many_to_one")
    if checked["left_party"].isna().any():
        raise ValueError("Forecast state missing from nominees")
    if not checked["left_party"].isin(["Democratic", "Independent"]).all():
        raise ValueError("Expected a Democratic or independent left-side candidate")
    if not (checked["party"].eq("Republican") | checked["party"].eq(checked["left_party"])).all():
        raise ValueError("Forecast party does not match the candidate table")
    if not (checked["winner"].eq("Republican") | checked["winner"].eq(checked["left_party"])).all():
        raise ValueError("Forecast winner does not match the candidate table")
    draws = checked.drop_duplicates(keys).copy()
    draws["left_win"] = draws["winner"].eq(draws["left_party"])
    summary = draws.groupby(["state", "model"]).agg(
        n_draws=("draw", "size"), p_left_win=("left_win", "mean")
    ).reset_index()
    summary["p_rep_win"] = 1 - summary["p_left_win"]
    return summary.merge(nominees[["state", "left_party"]], on="state", validate="many_to_one")


def main():
    forecast = config.read_table(config.FORECAST_FILE)
    nominees = pd.read_csv(config.NOMINEES_FILE)
    summary = summarize(forecast, nominees)
    models = sorted(summary["model"].unique())
    order = summary.groupby("state")["p_left_win"].mean().sort_values().index
    parties = nominees.set_index("state")["left_party"]
    y = np.arange(len(order))
    fig, axes = plt.subplots(
        1, len(models), figsize=(5.1 * len(models), 0.30 * len(order) + 2.3),
        sharey=True, squeeze=False,
    )
    for ax, model in zip(axes[0], models):
        values = summary[summary["model"] == model].set_index("state").reindex(order)
        available = values["p_left_win"].notna().to_numpy()
        left = values.loc[available, "p_left_win"].to_numpy() * 100
        colors = [BLUE if parties[state] == "Democratic" else INDEPENDENT
                  for state in order[available]]
        ax.barh(y[available], left, color=colors, height=0.72)
        ax.barh(y[available], 100 - left, left=left, color=RED, height=0.72)
        for pos, value in zip(y[available], left):
            if value >= 12:
                ax.text(value / 2, pos, f"{value:.0f}%", ha="center", va="center", color="white", fontsize=7)
            if 100 - value >= 12:
                ax.text((value + 100) / 2, pos, f"{100-value:.0f}%", ha="center", va="center", color="white", fontsize=7)
        for pos in y[~available]:
            ax.text(50, pos, "No forecast", ha="center", va="center", fontsize=7, color="gray")
        ax.axvline(50, color="black", lw=0.7, linestyle="--", alpha=0.5)
        ax.set(xlim=(0, 100), xlabel="Share of simulated wins (%)", title=model)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_yticks(y, [s + ("*" if parties[s] == "Independent" else "") for s in order], fontsize=8)
    axes[0, 0].invert_yaxis()
    fig.suptitle("Senate simulation: state win probabilities", fontsize=13)
    fig.legend(handles=[Patch(color=BLUE, label="Democratic"), Patch(color=INDEPENDENT, label="Independent*"),
                        Patch(color=RED, label="Republican")], loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=3, frameon=False)
    fig.text(0.02, 0.015, "* Independent opposition candidate. Each draw counts once.\n"
             "Simulated win frequencies, not vote shares; reliability depends on the forecast inputs.", fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    out = config.FIGURES
    out.mkdir(parents=True, exist_ok=True)
    config.RUN_RESULTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.RUN_RESULTS / "win_probabilities.csv", index=False)
    fig.savefig(out / "forecast_win_probability_bar.png", dpi=200)
    plt.close(fig)
    print(f"Saved {config.RUN_RESULTS / 'win_probabilities.csv'}")
    print(f"Saved {out / 'forecast_win_probability_bar.png'}")


if __name__ == "__main__":
    main()
