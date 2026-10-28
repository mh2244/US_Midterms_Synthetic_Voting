#### Preamble ####
# Purpose: Copies the files the website shows from one run into fixed "latest" folders, so
#          index.qmd never needs editing when a new run arrives, and writes a short line with
#          the run date and the headline numbers.
# Author: Kexin (Coco) Liu
# Date: 26 October 2026
# License: MIT
# Pre-requisites: Run scripts 11 (with the default --combine ivw last), 13, 13b, 15, 15b and 16 for this RUN.
# Any other information needed? Run from the project root:
#   python scripts/17-publish_latest.py            <- RUN from config / the RUN variable
#   RUN=2026-10-29 python scripts/17-publish_latest.py --force   <- publish an older run anyway
# Outputs:
#   figures/latest/seat_bar.png, forecast_dotplot.png, forecast_win_probability_bar.png
#   data/03-results/latest/  headline.md, headline.json, senate_map.md, crosstabs.md, win_probabilities.md, chamber_summary.csv, win_probabilities.csv,
#                            ivw_weights.csv, state_comparison.csv, run_info.md, run_info.json
# WHY the checks: the page must show the headline method (inverse-variance) from a news run, and
#   a rerun of an older day must not replace a newer one by accident.

#### Workspace setup ####
import argparse
import json
import shutil
import sys
from datetime import date

import pandas as pd

import config

ap = argparse.ArgumentParser()
ap.add_argument("--force", action="store_true", help="publish even if a newer run is already published")
args = ap.parse_args()

if config.RUN == "nonews":
    print("RUN=nonews is the comparison baseline; the website shows news runs. Nothing published.")
    sys.exit(0)

FIGS = ["seat_bar.png", "forecast_dotplot.png", "forecast_win_probability_bar.png"]
# Copied when present; the sensitivity table (16c) and the battleground summary (15c) are extra
# and must not block publishing
OPTIONAL = ["sensitivity.md", "sensitivity.csv", "results_summary.md", "battlegrounds.md", "battlegrounds.csv"]
OPTIONAL_FIGS = ["seat_distribution.png"]
TABLES = ["headline.md", "headline.json", "senate_map.md", "crosstabs.md", "win_probabilities.md", "chamber_summary.csv", "win_probabilities.csv", "ivw_weights.csv",
          "state_comparison.csv"]

#### Checks ####
missing = [str(config.FIGURES / f) for f in FIGS if not (config.FIGURES / f).exists()] + \
          [str(config.RUN_RESULTS / f) for f in TABLES if not (config.RUN_RESULTS / f).exists()]
if missing:
    sys.exit("Not published; missing files:\n  " + "\n  ".join(missing))

chamber = pd.read_csv(config.RUN_RESULTS / "chamber_summary.csv")
if "combine" not in chamber or set(chamber["combine"]) != {"ivw"}:
    sys.exit("Not published: chamber_summary.csv is not from the inverse-variance forecast. "
             "Run scripts/11-build_forecast.py with no --combine option last, then 13 and 15.")

info_file = config.LATEST_RESULTS / "run_info.json"
if info_file.exists() and not args.force:
    published = json.loads(info_file.read_text())["run"]
    if published > config.RUN:
        sys.exit(f"Not published: run {published} is newer than {config.RUN}. Use --force to replace it.")

#### Copy ####
config.LATEST_FIGURES.mkdir(parents=True, exist_ok=True)
config.LATEST_RESULTS.mkdir(parents=True, exist_ok=True)
for f in FIGS:
    shutil.copy2(config.FIGURES / f, config.LATEST_FIGURES / f)
for f in TABLES:
    shutil.copy2(config.RUN_RESULTS / f, config.LATEST_RESULTS / f)
for f in OPTIONAL:
    if (config.RUN_RESULTS / f).exists():
        shutil.copy2(config.RUN_RESULTS / f, config.LATEST_RESULTS / f)
for f in OPTIONAL_FIGS:
    if (config.FIGURES / f).exists():
        shutil.copy2(config.FIGURES / f, config.LATEST_FIGURES / f)

#### Run line for the page ####
p = chamber["p_rep_control"] * 100
seats = chamber["rep_median"]
day = date.fromisoformat(config.RUN)
when = f"{day.day} {day:%B %Y}"
pct_range = f"{p.min():.0f} per cent" if round(p.min()) == round(p.max()) else f"{p.min():.0f} to {p.max():.0f} per cent"
seat_range = f"{seats.min():.0f}" if seats.min() == seats.max() else f"{seats.min():.0f} to {seats.max():.0f}"
line = (f"Latest run: {when}, with that day's news. Across the {len(chamber)} models, the chance that "
        f"Republicans control the Senate is {pct_range}, and the median number of Republican seats "
        f"is {seat_range}.")
(config.LATEST_RESULTS / "run_info.md").write_text(line + "\n")
info_file.write_text(json.dumps({"run": config.RUN, "models": chamber["model"].tolist(),
                                 "p_rep_control": chamber["p_rep_control"].round(3).tolist(),
                                 "rep_median": chamber["rep_median"].tolist()}, indent=2) + "\n")

print(f"Published run {config.RUN} to {config.LATEST_FIGURES} and {config.LATEST_RESULTS}")
print(line)
