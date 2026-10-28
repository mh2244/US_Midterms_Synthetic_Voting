#### Preamble ####
# Purpose: Makes the interactive Senate map for the website: each state with a 2026 race is
#          coloured by the side the forecast favours and how strongly; hovering over a state shows
#          the candidates, the forecast vote share with a 90% interval, and the chance of winning.
#          The four models are pooled (each model's draws count equally), as in scripts 13 and 15b.
# Author: Kexin (Coco) Liu
# Date: 26 October 2026
# License: MIT
# Pre-requisites: Run scripts/02b-download_map.py once (state outlines) and
#   scripts/11-build_forecast.py (the default --combine ivw last).
# Any other information needed? Run from the project root:  python scripts/13b-make_map.py
# Outputs: data/03-results/run_<RUN>/senate_map.md, an HTML block (an SVG map with a few lines of
#   JavaScript for the hover box) that index.qmd includes after script 17 copies it to
#   data/03-results/latest/. Nothing is loaded from the internet when the page opens.
#
# HOW the map is drawn: the state outlines (longitude and latitude) are projected here with the
#   Albers USA projection used by d3 (Bostock): a conic equal-area projection for the lower 48,
#   with Alaska shrunk and Hawaii moved into insets at the bottom left.
# Colour rule (party colours from config.PARTY_COLOURS):
#   favoured side = the Democratic or independent candidate if they win at least half of the
#   simulations, otherwise the Republican. Strength = the favoured side's chance of winning:
#   85% or more = "likely" (full colour), 65% to 85% = "leans" (lighter), 55% to 65% = "tilts"
#   (lightest). Under 55% for both sides (45% to 55% each) = "toss-up" (sand), as forecasters
#   such as the Cook Political Report use a band rather than an exact 50-50. An independent favoured to win is grey at any
#   strength (the hover box gives the chance). States with no Senate race in 2026 are light grey;
#   their hover box names the two current senators. The legend always lists every category, even
#   those no state falls in this run.

#### Workspace setup ####
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import config

MAP_FILE = Path("data/01-raw_data/us-states.json")
# Senators of the 15 states with no race in 2026 (both seats not up), for the hover box only.
# Entered by hand from the Senate list on Wikipedia, checked 26 October 2026.
SENATORS_FILE = Path("data/01-raw_data/senators_not_up_2026.csv")
COL = config.PARTY_COLOURS
NO_RACE = config.NO_RACE_COLOUR
TOSS_UP = config.TOSS_UP_COLOUR     # neither side clearly favoured (45% to 55% each); not a party colour
LEVELS = [("Likely", 0.85, 0.0), ("Leans", 0.65, 0.45), ("Tilts", 0.55, 0.72)]   # name, floor, tint
WIDTH, HEIGHT = 960, 500


def tint(hex_colour, f):
    """Mix a colour with white: f = 0 keeps it, f = 1 gives white."""
    rgb = [int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (255 - c) * f):02x}" for c in rgb)


#### Albers USA projection (the d3 defaults for a 960 x 500 map) ####
def conic_equal_area(parallels, rotate, centre, scale, translate):
    p0, p1 = (math.radians(p) for p in parallels)
    n = (math.sin(p0) + math.sin(p1)) / 2
    c = 1 + math.sin(p0) * (2 * n - math.sin(p0))
    r0 = math.sqrt(c) / n

    def raw(lon, lat):
        lam = math.radians((lon + rotate + 180) % 360 - 180)
        r = math.sqrt(max(c - 2 * n * math.sin(math.radians(lat)), 0)) / n
        return r * math.sin(lam * n), r0 - r * math.cos(lam * n)

    # centre is given in the rotated frame, so project it without adding the rotation again
    cx, cy = raw(centre[0] - rotate, centre[1])

    def project(lon, lat):
        x, y = raw(lon, lat)
        return translate[0] + scale * (x - cx), translate[1] - scale * (y - cy)
    return project


K, TX, TY = 1070, 480, 250
PROJECT = {
    "lower48": conic_equal_area((29.5, 45.5), 96, (-0.6, 38.7), K, (TX, TY)),
    "Alaska": conic_equal_area((55, 65), 154, (-2, 58.5), 0.35 * K, (TX - 0.307 * K, TY + 0.201 * K)),
    "Hawaii": conic_equal_area((8, 18), 157, (-3, 19.9), K, (TX - 0.205 * K, TY + 0.212 * K)),
}


# Small north-eastern states get their label to the right, with a leader line
CALLOUT = ["VT", "NH", "MA", "RI", "CT", "NJ", "DE", "MD"]
NUDGE = {"FL": (14, 0), "MI": (8, 14), "LA": (-10, -3), "KY": (6, 2), "ID": (0, 14), "HI": (12, -2),
         "CA": (-8, 0), "WV": (-3, 3), "VA": (6, 2), "NY": (6, 0), "MN": (-6, 0), "OK": (10, 0)}


def label_point(geometry, project):
    """Area-weighted centroid of the largest projected polygon (the main body of the state)."""
    polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    best = None
    for polygon in polygons:
        pts = [project(lon, lat) for lon, lat in polygon[0]]
        a = cx = cy = 0.0
        for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
            cross = x0 * y1 - x1 * y0
            a += cross
            cx += (x0 + x1) * cross
            cy += (y0 + y1) * cross
        if a and (best is None or abs(a) > abs(best[0])):
            best = (a, cx / (3 * a), cy / (3 * a))
    return best[1], best[2]


def ink_for(fill):
    """White text on dark fills, near-black on light fills."""
    r, g, b = (int(fill[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r ** 2.2 + 0.7152 * g ** 2.2 + 0.0722 * b ** 2.2
    return "#ffffff" if lum < 0.3 else "#1a1a1a"


def svg_path(geometry, project):
    polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    parts = []
    for polygon in polygons:
        for ring in polygon:
            pts = [project(lon, lat) for lon, lat in ring]
            parts.append("M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + "Z")
    return "".join(parts)


#### State summaries ####
forecast = config.read_table(config.FORECAST_FILE)
nominees = pd.read_csv(config.NOMINEES_FILE).set_index("state")
left = forecast[forecast["party"] != "Republican"]


def pct(p):
    return "<1%" if p < 0.005 else ">99%" if p > 0.995 else f"{p * 100:.0f}%"


info = {}
for state, g in left.groupby("state"):
    race = nominees.loc[state]
    p_left = g.assign(w=g["winner"] != "Republican").groupby("model")["w"].mean().mean()
    lo, mid, hi = np.quantile(g["share"], [0.05, 0.5, 0.95]) * 100
    if max(p_left, 1 - p_left) < LEVELS[-1][1]:
        side, category, colour = None, "Toss-up", TOSS_UP
    else:
        side = race["left_party"] if p_left > 0.5 else "Republican"
        level, _, f = next(lv for lv in LEVELS if max(p_left, 1 - p_left) >= lv[1])
        category = "Independent favoured" if side == "Independent" else f"{level} {side}"
        colour = COL["Independent"] if side == "Independent" else tint(COL[side], f)
    l_last, r_last = race["left_name"].split()[-1], race["rep_name"].split()[-1]
    special = " (special election)" if race["race_type"] == "special" else ""
    info[state] = {
        "category": category, "colour": colour,
        "tip": (f"<b>{html.escape(config.STATE_NAME.get(state, state))}{special}</b>"
                f"{html.escape(race['left_name'])} ({race['left_party'][0]}) v. {html.escape(race['rep_name'])} (R)<br>"
                f"{html.escape(l_last)}'s share: {mid:.0f}% (90% interval {lo:.0f} to {hi:.0f})<br>"
                f"Chance {html.escape(l_last)} wins: {pct(p_left)}<br>"
                f"Chance {html.escape(r_last)} wins: {pct(1 - p_left)}"),
    }
assert len(info) == len(nominees), "A race is missing from the forecast"

#### SVG ####
ABBR = {"Democratic": "D", "Republican": "R"}
sen_table = pd.read_csv(SENATORS_FILE)
senators = {st: "<br>".join(f"{html.escape(r.senator)} ({ABBR.get(r.party, 'I')})" for r in g.itertuples())
            for st, g in sen_table.groupby("state")}
assert not set(senators) & set(info), "A state with a 2026 race is listed as having no race"
shapes = json.loads(MAP_FILE.read_text())["features"]
paths, labels = [], []
drawn = set()
callouts = {}
for feature in shapes:
    name = feature["properties"]["name"]
    code = config.STATE_ABBR.get(name)
    if code is None:                    # Puerto Rico and anything else that is not a state
        continue
    drawn.add(code)
    project = PROJECT.get(name, PROJECT["lower48"])
    if code in info:
        fill, tip = info[code]["colour"], info[code]["tip"]
    else:
        sen = senators.get(code)
        assert sen, f"No senators listed for {code} in {SENATORS_FILE}"
        fill, tip = NO_RACE, f"<b>{html.escape(name)}</b>No Senate race in 2026<br>Current senators:<br>{sen}"
    paths.append(f'<path d="{svg_path(feature["geometry"], project)}" fill="{fill}" '
                 f'data-tip="{html.escape(tip, quote=True)}" tabindex="0"><title>{html.escape(name)}</title></path>')
    x, y = label_point(feature["geometry"], project)
    if code in CALLOUT:
        callouts[code] = (x, y)
    else:
        dx, dy = NUDGE.get(code, (0, 0))
        labels.append(f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" fill="{ink_for(fill)}">{code}</text>')
# Callout labels in a column to the right of the coast, in north-to-south order
for i, code in enumerate(sorted(callouts, key=lambda c: callouts[c][1])):
    x, y = callouts[code]
    ly = 118 + i * 21
    labels.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="903" y2="{ly - 4}" class="map-leader"/>'
                  f'<text x="908" y="{ly}" fill="#1a1a1a" text-anchor="start">{code}</text>')
assert set(info) <= drawn, f"No outline for {sorted(set(info) - drawn)}"

# Fixed legend in three columns: Democratic and Republican from deep to light, then the rest.
# Every category is listed, even if no state falls in it this run.
columns = [("Democratic", [(lv, f"{lv} Democratic", tint(COL["Democratic"], f)) for lv, _, f in LEVELS]),
           ("Republican", [(lv, f"{lv} Republican", tint(COL["Republican"], f)) for lv, _, f in LEVELS]),
           ("Other", [("Toss-up", "Toss-up", TOSS_UP),
                      ("Independent favoured", "Independent favoured", COL["Independent"]),
                      ("No race in 2026", "No race", NO_RACE)])]
categories = {cat for _, items in columns for _, cat, _ in items}
assert {v["category"] for v in info.values()} <= categories, "A state's category is not in the legend"
swatches = "".join(
    f'<div class="map-col"><div class="map-col-head">{head}</div>'
    + "".join(f'<div class="map-key"><span class="map-swatch" style="background:{c}"></span>{label}</div>'
              for label, _, c in items) + "</div>"
    for head, items in columns)
day = "no news" if config.RUN == "nonews" else config.RUN
block = f'''```{{=html}}
<div class="senate-map">
<svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="Map of the 2026 Senate forecast by state">
{"".join(paths)}
<g class="map-labels">{"".join(labels)}</g>
</svg>
<div class="map-tip" hidden></div>
</div>
<div class="map-legend">{swatches}</div>
<p class="map-note">Hover over (or tap) a state for the candidates, the forecast share and the chance of
winning. Chance of winning for the favoured side: likely 85% or more, leans 65% to 85%, tilts 55% to 65%;
toss-up means each side has a 45% to 55% chance.
Four models pooled; run {day}.</p>
<style>
.senate-map {{ position: relative; }}
.senate-map svg {{ width: 100%; height: auto; display: block; }}
.senate-map path {{ stroke: #ffffff; stroke-width: 0.8; stroke-linejoin: round; cursor: pointer; }}
.senate-map path:hover, .senate-map path:focus {{ stroke: #1a1a1a; stroke-width: 1.6; outline: none; }}
.map-labels {{ pointer-events: none; font-family: var(--sans); font-size: 11px;
              font-weight: 600; text-anchor: middle; dominant-baseline: middle; }}
.map-labels text[text-anchor="start"] {{ text-anchor: start; }}
.map-leader {{ stroke: #888888; stroke-width: 0.7; }}
.map-tip {{ position: absolute; pointer-events: none; background: #ffffff; border: 1px solid #555555;
           padding: 0.45rem 0.6rem; font-family: var(--sans); font-size: 0.8rem;
           line-height: 1.4; color: #1a1a1a; max-width: 17rem; box-shadow: 0 2px 6px rgba(0,0,0,0.12); }}
.map-tip b {{ display: block; margin-bottom: 0.15rem; }}
.map-legend {{ display: flex; flex-wrap: wrap; gap: 0.6rem 2.5rem; justify-content: center;
              font-family: var(--sans); font-size: 0.8rem; margin-top: 0.5rem; }}
.map-col {{ display: flex; flex-direction: column; gap: 0.25rem; min-width: 8rem; }}
.map-col-head {{ font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.72rem;
                color: #1a1a1a; margin-bottom: 0.1rem; }}
.map-key {{ display: flex; align-items: center; gap: 0.4rem; }}
.map-swatch {{ width: 0.9rem; height: 0.9rem; border-radius: 2px; display: inline-block;
               box-shadow: inset 0 0 0 1px rgba(0,0,0,0.12); }}
.map-note {{ font-family: var(--sans); font-size: 0.8rem; color: #555555;
            text-align: center; margin-top: 0.8rem; }}
</style>
<script>
(function () {{
  var box = document.querySelector(".senate-map"), tip = box.querySelector(".map-tip");
  function show(e, el) {{
    tip.innerHTML = el.getAttribute("data-tip"); tip.hidden = false;
    var r = box.getBoundingClientRect(), x = (e.clientX || r.left + r.width / 2) - r.left,
        y = (e.clientY || r.top + r.height / 2) - r.top;
    var left = Math.min(x + 14, r.width - tip.offsetWidth - 4);
    tip.style.left = Math.max(left, 4) + "px"; tip.style.top = Math.max(y - tip.offsetHeight - 12, 4) + "px";
  }}
  box.querySelectorAll("path").forEach(function (el) {{
    el.addEventListener("mousemove", function (e) {{ show(e, el); }});
    el.addEventListener("click", function (e) {{ show(e, el); }});
    el.addEventListener("focus", function (e) {{ show(e, el); }});
    el.addEventListener("mouseleave", function () {{ tip.hidden = true; }});
    el.addEventListener("blur", function () {{ tip.hidden = true; }});
  }});
}})();
</script>
```
'''
config.RUN_RESULTS.mkdir(parents=True, exist_ok=True)
(config.RUN_RESULTS / "senate_map.md").write_text(block)
print(pd.Series({k: v["category"] for k, v in info.items()}).value_counts().to_string())
print(f"Saved {config.RUN_RESULTS / 'senate_map.md'} ({len(block) / 1000:.0f} kB)")
