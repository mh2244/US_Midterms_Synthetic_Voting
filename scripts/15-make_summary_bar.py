#### Preamble ####
# Purpose: Summarizes forecast draws as state-level win probabilities and plots
#          stacked horizontal bars for the Democratic/independent and Republican sides.
# Author: Jingxuan Feng
# Date: 26 October 2026
# Contact: amelia.f.feng@outlook.com
# License: MIT
# Pre-requisites: Run 11-build_forecast.py; install pandas numpy matplotlib pyarrow.
# Run from the project root: python scripts/15-make_summary_bar.py
# Outputs: data/03-results/run_<RUN>/win_probabilities.csv, win_probabilities.md (the table on
#          the website: one row per race, all models and each model) and
#          figures/run_<RUN>/forecast_win_probability_bar.png (kept for reference, not on the page)
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

BLUE, RED, INDEPENDENT = (config.PARTY_COLOURS[k] for k in ("Democratic", "Republican", "Independent"))


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


def pct_text(p):
    """Win chance as a whole per cent; simulated 0 and 100 are shown as <1 and >99."""
    v = p * 100
    return "<1" if v < 0.5 else ">99" if v > 99.5 else f"{v:.0f}"


def table_md(summary, nominees):
    """Markdown table: chance the Democratic or independent candidate wins, by race and model."""
    wide = summary.pivot(index="state", columns="model", values="p_left_win")
    models = [m for m in config.MODEL_LABELS if m in wide.columns] + \
             sorted(set(wide.columns) - set(config.MODEL_LABELS))
    wide["all"] = wide[models].mean(axis=1)            # each model counts equally
    wide = wide.sort_values("all", ascending=False)
    races = nominees.set_index("state")
    head = ["State", "Candidates (left v. Republican)", "All models"] + \
           [config.MODEL_LABELS.get(m, m) for m in models]
    lines = ["| " + " | ".join(head) + " |", "|---|---|" + "---:|" * (len(head) - 2)]
    for state, row in wide.iterrows():
        r = races.loc[state]
        left = r["left_name"] + (" (I)" if r["left_party"] == "Independent" else "")
        cells = [config.STATE_NAME.get(state, state), f"{left} v. {r['rep_name']}",
                 f"**{pct_text(row['all'])}**"] + [pct_text(row[m]) for m in models]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    forecast = config.read_table(config.FORECAST_FILE)
    nominees = pd.read_csv(config.NOMINEES_FILE)
    summary = summarize(forecast, nominees)
    models = [m for m in config.MODEL_LABELS if m in set(summary["model"])] + \
        sorted(set(summary["model"]) - set(config.MODEL_LABELS))
    order = summary.groupby("state")["p_left_win"].mean().sort_values(ascending=False).index
    parties = nominees.set_index("state")["left_party"]
    labels = [config.STATE_NAME.get(s, s) + ("*" if parties[s] == "Independent" else "") for s in order]
    y = np.arange(len(order))
    # Two models per row (larger than four side by side); states in the same order in every panel,
    # most Democratic at the top
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    ncol = 2
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, nrow * (0.22 * len(order) + 1.0) + 0.8),
                             sharex=True, squeeze=False)
    for ax, model in zip(axes.flat, models):
        values = summary[summary["model"] == model].set_index("state").reindex(order)
        available = values["p_left_win"].notna().to_numpy()
        left = values.loc[available, "p_left_win"].to_numpy() * 100
        colors = [BLUE if parties[state] == "Democratic" else INDEPENDENT for state in order[available]]
        ax.barh(y[available], left, color=colors, height=0.75)
        ax.barh(y[available], 100 - left, left=left, color=RED, height=0.75)
        for pos, value in zip(y[available], left):
            if value >= 10:
                ax.text(value / 2, pos, f"{value:.0f}", ha="center", va="center", color="white", fontsize=7.5)
            if 100 - value >= 10:
                ax.text((value + 100) / 2, pos, f"{100 - value:.0f}", ha="center", va="center", color="white",
                        fontsize=7.5)
        for pos in y[~available]:
            ax.text(50, pos, "No forecast", ha="center", va="center", fontsize=7.5, color="gray")
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlim(0, 100)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_title(config.MODEL_LABELS.get(model, model), loc="left", fontsize=11, fontweight="bold")
        ax.set_yticks(y, labels, fontsize=8.5)
        ax.tick_params(axis="y", length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Share of simulated outcomes won (%)", fontsize=9)
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    fig.legend(handles=[Patch(color=BLUE, label="Democratic"), Patch(color=INDEPENDENT, label="Independent*"),
                        Patch(color=RED, label="Republican")], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=9.5)
    fig.text(0.01, 0.005, "* Independent candidate. Numbers in the bars are per cent of simulated outcomes. "
             f"Run {config.RUN}.", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.015, 1, 0.975), h_pad=2.0, w_pad=2.5)
    out = config.FIGURES
    out.mkdir(parents=True, exist_ok=True)
    config.RUN_RESULTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.RUN_RESULTS / "win_probabilities.csv", index=False)
    (config.RUN_RESULTS / "win_probabilities.md").write_text(table_md(summary, nominees))
    fig.savefig(out / "forecast_win_probability_bar.png", dpi=200)
    plt.close(fig)
    print(f"Saved {config.RUN_RESULTS / 'win_probabilities.csv'}")
    print(f"Saved {out / 'forecast_win_probability_bar.png'}")


if __name__ == "__main__":
    main()
