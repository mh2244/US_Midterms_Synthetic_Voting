#### Preamble ####
# Purpose: Sets our forecast beside a prediction market: for each news day, our chance that the
#          Democratic or independent candidate wins each race (four models pooled) against the
#          PredictIt price on the same day, plus the chance of Senate control. Following Wang,
#          Rothschild, Goel and Gelman (2015), who compared their state forecasts with Betfair and
#          Intrade because there is no ground truth before the election.
# Author: Kexin (Coco) Liu
# Date: 29 October 2026
# License: MIT
# Pre-requisites: scripts/02c-download_predictit.py (within 7 days of each news day) and script 11
#   for the news days.
# Any other information needed? Run from the project root:  python scripts/16f-compare_markets.py
# Outputs: data/03-results/comparison/markets.csv and markets.md (summary and table), and
#   figures/run_<latest news day>/market_comparison.png (script 17 copies it to figures/latest).
#
# HOW a price becomes a probability: a PredictIt contract pays $1 if it comes true, so its price is
#   roughly the market's probability. The prices of the two sides add up to a little more than 1
#   (the market's margin), so the left side's price is divided by the sum of the two sides' prices.
#   Contracts for anyone else (a third candidate) are left out, as in our two-sided forecast.
# The market is not ground truth and not independent of the polls, which traders also read; it is
#   a forecast that combines the polls with other information through trading.

#### Workspace setup ####
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

NEWS_DAYS = ["2026-10-27", "2026-10-29", "2026-10-31"]
MARKETS = Path("data/01-raw_data/markets")
OUT = config.RESULTS / "comparison"
LABEL_GAP = 0.08                          # label states where the two differ by 8 points or more


def day(run):
    d = pd.Timestamp(run)
    return f"{d.day} {d:%B}"


#### Market prices: every download, newest kept for each day ####
files = sorted(MARKETS.glob("predictit_*/daily_prices.csv"))
assert files, "No PredictIt download yet: run scripts/02c-download_predictit.py"
prices = pd.concat([pd.read_csv(f).assign(download=f.parent.name) for f in files], ignore_index=True)
prices = prices.sort_values("download").drop_duplicates(["market_id", "contract_id", "date"], keep="last")

nominees = pd.read_csv(config.NOMINEES_FILE).set_index("state")


def side(state, name):
    """Which side a contract is on: left (the Democrat, or the independent where there is no
    Democrat), Republican, or other."""
    n = name.lower()
    if state == "control":
        return "left" if "dem" in n else "rep" if "rep" in n else "other"
    race = nominees.loc[state]
    if config.last_name(race["left_name"]) in n:
        return "left"
    if config.last_name(race["rep_name"]) in n or "rep" in n or "gop" in n:
        return "rep"
    if ("dem" in n and race["left_party"] == "Democratic") or ("indep" in n and race["left_party"] == "Independent"):
        return "left"
    return "other"


prices["side"] = [side(s, c) for s, c in zip(prices["state"], prices["contract_name"])]
two = prices[prices["side"] != "other"].pivot_table(index=["state", "date"], columns="side", values="close", aggfunc="sum")
two["market_p_left"] = two["left"] / (two["left"] + two["rep"])
two = two.reset_index()

#### Our forecast for each news day ####
rows, summary = [], []
for run in NEWS_DAYS:
    path = config.RESULTS / f"run_{run}" / "forecast.parquet"
    m = two[two["date"] == run].set_index("state")
    if not path.exists() or m.empty:
        continue
    f = config.read_table(path)
    left = f[f["party"] != "Republican"].assign(win=lambda d: d["winner"] != "Republican")
    ours = left.groupby(["state", "model"])["win"].mean().groupby("state").mean()
    races = m.drop(index="control", errors="ignore").join(ours.rename("ours_p_left"), how="inner")
    races["difference"] = races["ours_p_left"] - races["market_p_left"]
    races = races.assign(run=run).reset_index()[["run", "state", "market_p_left", "ours_p_left", "difference"]]
    rows.append(races)
    chamber = pd.read_csv(config.RESULTS / f"run_{run}" / "chamber_summary.csv")
    differs = races[(races["ours_p_left"] > 0.5) != (races["market_p_left"] > 0.5)]["state"]
    summary.append({
        "run": run, "n": len(races),
        "corr": races["ours_p_left"].corr(races["market_p_left"]),
        "mad": races["difference"].abs().mean(),
        "differs": ", ".join(config.STATE_NAME.get(s, s) for s in differs) or "none",
        "ours_dem_control": 1 - chamber["p_rep_control"].mean(),
        "market_dem_control": m.loc["control", "market_p_left"] if "control" in m.index else np.nan,
    })
assert rows, "No news day has both a forecast and a PredictIt price yet"
table = pd.concat(rows, ignore_index=True)
summ = pd.DataFrame(summary)
OUT.mkdir(parents=True, exist_ok=True)
table.round(4).to_csv(OUT / "markets.csv", index=False)


#### Table for the website ####
def signed(x):
    """A difference in points with a typographic minus sign; 0 when it rounds to zero."""
    v = round(x * 100)
    return "0" if v == 0 else f"{v:+d}".replace("-", "\u2212")


def pct(p):
    return "<1" if p < 0.005 else ">99" if p > 0.995 else f"{p * 100:.0f}"


md = ["| News day | States | Correlation | Mean absolute difference (points) | Favourite differs | "
      "Democratic control: ours / market (%) |", "|---|---:|---:|---:|---|---:|"]
for s in summ.itertuples():
    ctrl = f"{s.ours_dem_control * 100:.0f} / {s.market_dem_control * 100:.0f}" if pd.notna(s.market_dem_control) else "–"
    md.append(f"| {day(s.run)} | {s.n} | {s.corr:.2f} | {s.mad * 100:.1f} | {s.differs} | {ctrl} |")
latest = summ["run"].iloc[-1]
t = table[table["run"] == latest].sort_values("difference")
md += ["", "<details>", f"<summary>Table view: every race, {day(latest)}</summary>", "",
       "| State | Market (%) | Ours (%) | Ours minus market (points) |", "|---|---:|---:|---:|"]
for r in t.itertuples():
    name = config.STATE_NAME.get(r.state, r.state) + (" (I)" if nominees.loc[r.state, "left_party"] == "Independent" else "")
    md.append(f"| {name} | {pct(r.market_p_left)} | {pct(r.ours_p_left)} | {signed(r.difference)} |")
md += ["", "</details>"]
(OUT / "markets.md").write_text("\n".join(md) + "\n")

#### Figure: ours against the market, latest news day ####
config.use_site_style()
fig, ax = plt.subplots(figsize=(6.2, 6.2))
ax.fill_between([0, 0.5], 0.5, 1, color="#f2f2f2", lw=0, zorder=0)
ax.fill_between([0.5, 1], 0, 0.5, color="#f2f2f2", lw=0, zorder=0)
ax.text(0.02, 0.98, "We favour the left side,\nthe market the Republican", va="top", fontsize=8, color=config.MUTED)
ax.text(0.98, 0.02, "We favour the Republican,\nthe market the left side", ha="right", fontsize=8, color=config.MUTED)
ax.plot([0, 1], [0, 1], color=config.INK, lw=1.2, ls=(0, (3, 2)), zorder=1)
col = [config.PARTY_COLOURS["Independent"] if nominees.loc[s, "left_party"] == "Independent"
       else config.PARTY_COLOURS["Democratic"] for s in t["state"]]
ax.scatter(t["market_p_left"], t["ours_p_left"], s=36, color=col, edgecolor="white", lw=0.6, zorder=3)
placed = []                               # (x, y, side) of labels so far, to keep them apart


def clash(x, y, ha):
    left = x - 0.2 if ha == "right" else x
    return any(abs(y - py) < 0.03 and left < px2 and px1 < left + 0.2 for px1, px2, py in placed)


pts = list(zip(t["market_p_left"], t["ours_p_left"]))
for r in t[t["difference"].abs() >= LABEL_GAP].sort_values("ours_p_left", ascending=False).itertuples():
    px, py = r.market_p_left, r.ours_p_left
    for ha, x in (("left", px + 0.015), ("right", px - 0.015)):
        near_point = any(abs(py - qy) < 0.03 and (0 < qx - px < 0.2 if ha == "left" else 0 < px - qx < 0.2)
                         for qx, qy in pts)
        if not clash(x, py, ha) and not near_point:
            break
    else:                                 # both sides taken: right side, shifted down
        ha, x = "left", px + 0.015
    y = py
    while clash(x, y, ha):
        y -= 0.03
    placed.append(((x - 0.2, x) if ha == "right" else (x, x + 0.2)) + (y,))
    ax.annotate(config.STATE_NAME.get(r.state, r.state), (px, py), xytext=(x, y), fontsize=8, ha=ha,
                va="center", color=config.INK,
                arrowprops=None if abs(y - py) < 0.005 else {"arrowstyle": "-", "lw": 0.5, "color": config.MUTED})
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_xticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
ax.set_xlabel("PredictIt: chance the Democratic or independent candidate wins")
ax.set_ylabel("Our forecast: chance the Democratic or independent candidate wins")
ax.set_aspect("equal")
fig.tight_layout()
fig_dir = Path("figures") / f"run_{latest}"
fig_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_dir / "market_comparison.png")
plt.close(fig)

#### Checks ####
assert table[["market_p_left", "ours_p_left"]].stack().between(0, 1).all()
assert table.groupby("run")["state"].nunique().le(len(nominees)).all()
print("\n".join(md))
print(f"Saved {OUT / 'markets.md'} and {fig_dir / 'market_comparison.png'}")
