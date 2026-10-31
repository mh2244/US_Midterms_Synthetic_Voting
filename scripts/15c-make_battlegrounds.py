#### Preamble ####
# Purpose: The summary under the headline: a short results line (seats that change hands and
#          the toss-ups), a table of the battleground races with the party that holds each seat
#          now and whether the forecast has it changing hands, and the distribution of Republican
#          seats across all simulations. Four models pooled, as in scripts 13, 13b and 15b.
# Author: Kexin (Coco) Liu
# Date: 27 October 2026
# License: MIT
# Pre-requisites: Run scripts/11-build_forecast.py (the default --combine ivw last).
# Any other information needed? Run from the project root:  python scripts/15c-make_battlegrounds.py
# Inputs: data/01-raw_data/senate_seats_up_2026.csv (party holding each seat up in 2026, entered
#   by hand from the list of current senators on Wikipedia, checked 27 October 2026; 13 Democratic
#   and 22 Republican, as in script 11).
# Outputs: data/03-results/run_<RUN>/results_summary.md, battlegrounds.md, battlegrounds.csv and
#   figures/run_<RUN>/seat_distribution.png (script 17 copies them to the latest folders).
#
# Battlegrounds: the 12 races Ballotpedia lists as battlegrounds (Ballotpedia 2026).
# Categories use the same cut-offs as the map (script 13b): the favoured side's chance of winning
# is 85% or more = likely, 65% to 85% = leans, 55% to 65% = tilts; under 55% for both = toss-up.
# A seat is "forecast to change hands" (Flip = Yes) when the favoured side is not the party holding it now;
# a toss-up has no favoured side, so its Flip is shown as a dash.

#### Workspace setup ####
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

BATTLEGROUNDS = ["AK", "FL", "GA", "IA", "ME", "MI", "MT", "NE", "NH", "NC", "OH", "TX"]
SEATS_FILE = Path("data/01-raw_data/senate_seats_up_2026.csv")
LEVELS = [("Likely", 0.85), ("Leans", 0.65), ("Tilts", 0.55)]
DEM_NOT_UP, REP_NOT_UP = 34, 31          # as in script 11
COL = config.PARTY_COLOURS
SHORT = {"Democratic": "D", "Republican": "R", "Independent": "I"}


def pct(p):
    return "<1%" if p < 0.005 else ">99%" if p > 0.995 else f"{p * 100:.0f}%"


def names(states):
    names_ = [config.STATE_NAME.get(s, s) for s in states]
    return names_[0] if len(names_) == 1 else ", ".join(names_[:-1]) + " and " + names_[-1]


#### Per-state forecast ####
forecast = config.read_table(config.FORECAST_FILE)
nominees = pd.read_csv(config.NOMINEES_FILE).set_index("state")
holder = pd.read_csv(SEATS_FILE).set_index("state")["holder_party"]
assert set(holder.index) == set(nominees.index), "The seats file and the nominees do not cover the same races"
left = forecast[forecast["party"] != "Republican"]
rows = []
for state, g in left.groupby("state"):
    race = nominees.loc[state]
    p_left = g.assign(w=g["winner"] != "Republican").groupby("model")["w"].mean().mean()
    lo, mid, hi = np.quantile(g["share"], [0.05, 0.5, 0.95]) * 100
    p_side = max(p_left, 1 - p_left)
    if p_side < LEVELS[-1][1]:
        side, rating = None, "Toss-up"
    else:
        side = race["left_party"] if p_left > 0.5 else "Republican"
        rating = f"{next(n for n, f in LEVELS if p_side >= f)} {side}"
    now = holder[state]
    change = "" if side is None or side == now else f"Flip to {SHORT[side]}"
    rows.append({"state": state, "name": config.STATE_NAME.get(state, state), "holder": now,
                 "left_name": race["left_name"], "left_party": race["left_party"], "rep_name": race["rep_name"],
                 "share_mid": mid, "share_lo": lo, "share_hi": hi, "p_left": p_left,
                 "rating": rating, "change": change})
states = pd.DataFrame(rows).set_index("state")
assert len(states) == len(nominees), "A race is missing from the forecast"

#### Seats ####
results = forecast[["model", "draw", "state", "winner"]].drop_duplicates()
seats = (results.assign(rep=results["winner"] == "Republican").groupby(["model", "draw"])["rep"].sum()
         + REP_NOT_UP)
dist = seats.value_counts(normalize=True).sort_index()

#### Results line ####
flips = states[states["change"] != ""].sort_values("p_left", ascending=False)
toss = states[states["rating"] == "Toss-up"].sort_values("p_left", ascending=False)
to_d = flips[flips["change"] == "Flip to D"]
to_r = flips[flips["change"] == "Flip to R"]
to_i = flips[flips["change"] == "Flip to I"]
parts = []
if len(to_d):
    parts.append(f"Democrats are favoured to gain {names(to_d.index)}, now held by Republicans")
if len(to_i):
    parts.append(f"an independent is favoured in {names(to_i.index)}")
if len(to_r):
    parts.append(f"Republicans are favoured to gain {names(to_r.index)}, now held by Democrats")
line = ("; ".join(parts) + ". ") if parts else "No seat is favoured to change hands. "
if len(toss):
    line += (f"{names(toss.index)} {'is a toss-up' if len(toss) == 1 else 'are toss-ups'}, with each side "
             f"given a winning probability of 45% to 55%. ")
line = line.rstrip()                      # the chance of control is already in the headline (15b)
p_rep = (seats >= 50).mean()              # used for the seat-distribution label below
config.RUN_RESULTS.mkdir(parents=True, exist_ok=True)
(config.RUN_RESULTS / "results_summary.md").write_text(line + "\n")

#### Battleground table ####
bg = states.loc[BATTLEGROUNDS].sort_values("p_left", ascending=False)
bg.round(4).to_csv(config.RUN_RESULTS / "battlegrounds.csv")
md = ["| State | Seat now | Republican share, median (90% interval) | "
      "Republican winning probability | Rating | Flip |", "|:---:|:---:|:---:|:---:|:---:|:---:|"]   # all centred
for st, r in bg.iterrows():
    name = r["name"] + (" (I)" if r["left_party"] == "Independent" else "")
    flip = "–" if r["rating"] == "Toss-up" else ("Yes" if r["change"] else "No")
    md.append(f"| {name} | {SHORT[r['holder']]} | {100 - r['share_mid']:.0f}% ({100 - r['share_hi']:.0f} to {100 - r['share_lo']:.0f}) | "
              f"{pct(1 - r['p_left'])} | {r['rating']} | {flip} |")
(config.RUN_RESULTS / "battlegrounds.md").write_text("\n".join(md) + "\n")

#### Seat distribution ####
config.use_site_style()            # shared graph theme: scripts/site.mplstyle
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.bar(dist.index, dist.values * 100, width=0.82,
       color=[COL["Republican"] if s >= 50 else COL["Democratic"] for s in dist.index])
ax.axvline(49.5, color=config.INK, lw=1.2, ls=(0, (3, 2)))
top = dist.max() * 100
# Each label sits at the centre of its side of the 50-seat line, over the bars it describes.
lo, hi = dist.index.min() - 0.5, dist.index.max() + 0.5
ax.text((lo + 49.5) / 2, top * 1.04, f"Republicans lose control\n(49 seats or fewer): {(seats <= 49).mean() * 100:.0f}%",
        ha="center", va="bottom", fontsize=9, color=COL["Democratic"])
ax.text((49.5 + hi) / 2, top * 1.04, f"Republicans keep control\n(50 seats or more): {p_rep * 100:.0f}%",
        ha="center", va="bottom", fontsize=9, color=COL["Republican"])
ax.set_ylim(0, top * 1.35)
ax.set_xlabel("Republican seats in the next Senate", fontsize=9)
ax.set_ylabel("Share of simulations (%)", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=8.5)
fig.tight_layout()
config.FIGURES.mkdir(parents=True, exist_ok=True)
fig.savefig(config.FIGURES / "seat_distribution.png", dpi=200)
plt.close(fig)

#### Checks ####
assert np.isclose(dist.sum(), 1)
assert states["p_left"].between(0, 1).all()
assert ((seats >= REP_NOT_UP) & (seats <= REP_NOT_UP + len(nominees))).all()
print(line)
print("\n".join(md))
print(f"Saved results_summary.md, battlegrounds.md and {config.FIGURES / 'seat_distribution.png'}")
