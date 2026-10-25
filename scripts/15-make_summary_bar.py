#### Preamble ####
# Purpose: Builds the headline block for the top of the website: one sentence on chamber control
#          and a seat bar (Democratic seats not up, Democratic wins, independent wins, Republican
#          wins, Republican seats not up). Written as a small file that index.qmd includes.
# Author: (group)
# Date: 25 October 2026
# License: MIT
# Pre-requisites: Run scripts/11-build_forecast.py first. No packages beyond pandas and numpy.
# Any other information needed? Run from the project root:
#   python scripts/15-make_summary_bar.py
#   python scripts/15-make_summary_bar.py --note "Test run with 5 personas per state."
# Output: includes/forecast_summary.md  (index.qmd has {{< include includes/forecast_summary.md >}})
#   The file is an HTML block inside a ```{=html} fence, so Quarto does not read the indented
#   lines as code.
#
# HOW the numbers are made: for every draw of every model we count the races the Democrat wins,
# the races an independent wins, and the races the Republican wins. The bar shows the MEDIAN of
# each count over all draws and models (models count equally). The sentence reports how often
# the Democrats reach 51 seats, as a count out of 100 draws, not a percentage with decimals.
# LIMITS to write about:
#   - 51 seats is the majority. With exactly 50 the Vice President breaks ties, and in 2026 that
#     is a Republican. An independent who wins is NOT counted as a Democratic seat here.
#   - The bar uses medians, so the three counts always add to the number of races even when
#     individual draws differ a lot. The 80% range is printed under the bar.
#   - DEM_NOT_UP and REP_NOT_UP are copied from the group's simulation script. Check them: they
#     must add up to 100 minus the number of races (35).

#### Workspace setup ####
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config

ap = argparse.ArgumentParser()
ap.add_argument("--dem-not-up", type=int, default=34)
ap.add_argument("--rep-not-up", type=int, default=31)
ap.add_argument("--note", default="", help="short line under the bar, e.g. 'Test run: not a forecast.'")
ap.add_argument("--updated", default=config.AS_OF, help="date shown on the page (polls and news as of)")
args = ap.parse_args()

BLUE, RED, GREY_BLUE = "#2166ac", "#b2182b", "#5b7f95"
DARK_BLUE, DARK_RED = "#14385f", "#6e1020"     # seats not up for election
OUT = Path("includes")
OUT.mkdir(exist_ok=True)

#### Counts per draw ####
forecast = config.read_table(config.FORECAST_FILE)
nominees = pd.read_csv(config.NOMINEES_FILE)
n_races = nominees["state"].nunique()
assert args.dem_not_up + args.rep_not_up + n_races == 100, \
    f"Seats not up ({args.dem_not_up} + {args.rep_not_up}) plus {n_races} races must equal 100"

results = forecast[["model", "draw", "state", "winner"]].drop_duplicates()
by_draw = (results.assign(dem=results["winner"] == "Democratic",
                          ind=results["winner"] == "Independent",
                          rep=results["winner"] == "Republican")
           .groupby(["model", "draw"])[["dem", "ind", "rep"]].sum())
assert (by_draw.sum(axis=1) == n_races).all(), "A draw does not have one winner per race"

dem = int(round(by_draw["dem"].median()))
ind = int(round(by_draw["ind"].median()))
rep = n_races - dem - ind                      # keeps the bar at exactly 100 seats
dem_total = args.dem_not_up + by_draw["dem"]
p_control = float((dem_total >= 51).mean())
per_100 = int(round(p_control * 100))
lo, hi = (int(x) for x in np.percentile(dem_total, [10, 90]))
n_draws = by_draw.groupby("model").size().iloc[0]
n_models = by_draw.index.get_level_values("model").nunique()

#### Sentence ####
if per_100 >= 50:
    headline = f"Democrats win control of the Senate in {per_100} of 100 simulated outcomes."
else:
    headline = f"Republicans keep control of the Senate in {100 - per_100} of 100 simulated outcomes."

#### HTML ####
def seg(seats, color, label):
    return (f'<div class="seg" style="width:{seats}%;background:{color}" title="{label}: {seats}"></div>'
            if seats > 0 else "")


dem_all, rep_all = args.dem_not_up + dem, args.rep_not_up + rep
ind_label = f'<div class="mid"><span class="num" style="color:{GREY_BLUE}">{ind}</span> IND</div>' if ind else ""
bar = (seg(args.dem_not_up, DARK_BLUE, "Democratic seats not up") + seg(dem, BLUE, "Democratic wins")
       + seg(ind, GREY_BLUE, "Independent wins") + seg(rep, RED, "Republican wins")
       + seg(args.rep_not_up, DARK_RED, "Republican seats not up"))
alt = (f"Seat bar. Democrats {dem_all} seats ({args.dem_not_up} not up for election), "
       f"Republicans {rep_all} seats ({args.rep_not_up} not up)"
       + (f", independents {ind}" if ind else "") + ". Median of the simulated outcomes.")
note = f'<p class="bar-note">{args.note}</p>' if args.note else ""

html = f"""```{{=html}}
<div class="forecast-summary">
  <p class="headline">{headline}</p>
  <div class="bar" role="img" aria-label="{alt}">
    {bar}
    <div class="line50" title="51 seats is a majority"></div>
  </div>
  <div class="bar-labels">
    <div class="left"><span class="num" style="color:{BLUE}">{dem_all}</span> DEM<br><small>{args.dem_not_up} seats not up</small></div>
    {ind_label}
    <div class="right">REP <span class="num" style="color:{RED}">{rep_all}</span><br><small>{args.rep_not_up} seats not up</small></div>
  </div>
  <p class="bar-note">Median seats over {n_models * n_draws:,} simulated outcomes ({n_models} models x {n_draws:,} draws).
  In 80 of 100 outcomes the Democrats hold between {lo} and {hi} seats. Dashed line: 50 seats; 51 is a majority.
  Seats not up for election are held constant.</p>
  {note}
</div>
```
"""
(OUT / "forecast_summary.md").write_text(html)

print(f"Saved {OUT / 'forecast_summary.md'}")
print(headline)
print(f"Median Democratic wins {dem}, independent wins {ind}, Republican wins {rep} (of {n_races} races)")
print(f"Democratic total {dem_all}, Republican total {rep_all}; 80% range of Democratic seats {lo}-{hi}")
