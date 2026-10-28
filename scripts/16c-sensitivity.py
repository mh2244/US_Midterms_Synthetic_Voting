#### Preamble ####
# Purpose: Sensitivity of the headline to the two error sizes we set by assumption rather than
#          estimate: the national error shared by all states (config.NATIONAL_SD, 0.03) and the
#          state poll error beyond sampling (--poll-sd in script 11, 0.03). Reruns the
#          inverse-variance forecast of script 11 on a grid of both and tabulates the chance that
#          Republicans keep the Senate and the chance that Democrats win 51 seats on their own.
# Author: Kexin (Coco) Liu
# Date: 27 October 2026
# License: MIT
# Pre-requisites: Run scripts 10 and 11 for this RUN (needs scores.parquet and the polls file).
# Any other information needed? Run from the project root:  python scripts/16c-sensitivity.py
#   Each grid point calls script 11 with --summary-only, so the run's own forecast files are not
#   touched. No model is called; it takes a few minutes.
# Outputs (in data/03-results/run_<RUN>/): sensitivity.csv (one row per model and grid point)
#   and sensitivity.md (the table on the website, four models pooled).
# NOTE: the four models have the same number of draws, so pooling them is the plain mean of the
#   four per-model chances.

#### Workspace setup ####
import os
import subprocess
import sys

import pandas as pd

import config

NATIONAL = [0.0, 0.02, 0.03, 0.04]
POLL = [0.02, 0.03, 0.04]
OUT_CSV = config.RUN_RESULTS / "sensitivity.csv"
OUT_MD = config.RUN_RESULTS / "sensitivity.md"

#### Run the grid ####
if OUT_CSV.exists():
    OUT_CSV.unlink()                          # start fresh; script 11 appends to this file
env = {**os.environ, "RUN": config.RUN}
for nat in NATIONAL:
    for poll in POLL:
        print(f"national SD {nat}, poll SD {poll}", flush=True)
        subprocess.run([sys.executable, "scripts/11-build_forecast.py", "--combine", "ivw",
                        "--national-sd", str(nat), "--poll-sd", str(poll), "--summary-only", str(OUT_CSV)],
                       check=True, env=env, stdout=subprocess.DEVNULL)

#### Table ####
grid = pd.read_csv(OUT_CSV)
assert len(grid) == len(NATIONAL) * len(POLL) * grid["model"].nunique(), "A grid point is missing"
pooled = grid.groupby(["national_sd", "poll_sd"])[["p_rep_control", "p_dem_51_without_ind"]].mean()
assert pooled.stack().between(0, 1).all()


def cell(nat, poll):
    r = pooled.loc[(nat, poll)]
    text = f"{r['p_rep_control'] * 100:.0f} / {r['p_dem_51_without_ind'] * 100:.0f}"
    return f"**{text}**" if (nat, poll) == (config.NATIONAL_SD, 0.03) else text


md = ["| National error SD | " + " | ".join(f"Poll error {p:.2f}" for p in POLL) + " |",
      "|---|" + "---:|" * len(POLL)]
for nat in NATIONAL:
    md.append(f"| {nat:.2f} | " + " | ".join(cell(nat, p) for p in POLL) + " |")
OUT_MD.write_text("\n".join(md) + "\n")
print(f"Saved {OUT_CSV} and {OUT_MD}")
print(OUT_MD.read_text())
