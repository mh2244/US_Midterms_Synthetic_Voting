#### Preamble ####
# Purpose: Turns the Wikipedia candidate list into one row per race that names the main
#          "left" candidate (the Democrat, or the leading independent where there is none)
#          and the Republican. Later scripts put these names in the prompts and anchors. 
#          While there are multiple parties (Republican, Democrat, Independent, Liberterian, etc.)
#          this script compares the senate_candidates_2026.csv to the real polling data to identify
#          2 sides--SSR requires 1 scale (not Republican to definitely Republican, in our case). from
#          real polling data, idnetifies the Republican candidate with most vote share and opposition 
#          /non-Republican with highest vote share as the two main candidates. 
#           Would be a limitation to mention in our analysis.
# Author: Maggie, Coco, Amelia.
# Date: 22 October 2026
# License: MIT
# Pre-requisites: Run scripts/05-collect_candidates.py first. The poll file is optional,
#   but it helps pick the right person when a party lists several names.
# Any other information needed? Run from the project root. READ THE PRINTED WARNINGS and fix
#   the OVERRIDES table below by hand: Wikipedia lists are messy (see the notes there).
#
# WHY this script exists: the raw list has full state names, a "DFL" label in Minnesota, three
# Republicans in Alaska, no Democrat in Idaho and South Dakota, and an independent (Osborn) as
# the main challenger in Nebraska. Prompts need exactly one left and one right name.


#### Workspace setup ####
import pandas as pd

import config

# Fix by hand after checking Wikipedia / Ballotpedia. Keys: state code. Values: names to use.
# Alaska: the list splits the incumbent into "Dan J. Sullivan" and "Dan S. Sullivan"; the polls
# call him "Dan Sullivan". Nebraska: polls label Dan Osborn (Independent) as the main challenger.
# Idaho: the polls show David Roth as the Democrat, but he is missing from our candidate file;
# CHECK THIS before the final run (it is left to the rule below until you do).
# Montana (fixed 31 October): the rule above picks the Democrat whenever there is one, but here the
# main challenger is the independent Seth Bodnar (about 31 per cent in polls since mid-September,
# against 16 for the Democrat Alani Bankhead and 42 for the Republican Kurt Alme). The runs of 27
# and 29 October used Bankhead; from 31 October Bodnar is the left candidate and Bankhead an other.
OVERRIDES = {
    "AK": {"rep_name": "Dan Sullivan"},   # the name the polls use; this row was already in the file
    # "NE": {"left_name": "Dan Osborn", "left_party": "Independent"},
    "MT": {"left_name": "Seth Bodnar", "left_party": "Independent"},
}


last_name = config.last_name


def poll_mentions():
    """How often each last name appears in recent polls, by state. Empty if no poll file."""
    if not config.POLLS_FILE.exists():
        return {}
    polls = pd.read_csv(config.POLLS_FILE, encoding="utf-8-sig")
    polls = polls[polls["end_date"] >= "2026-08-01"]
    counts = {}
    for _, row in polls.iterrows():
        key = (row["state"], last_name(row["response_name"]))
        counts[key] = counts.get(key, 0) + 1
    return counts


def pick(frame, state, counts):
    """From several candidates for one side, take the one the polls mention most (else the first)."""
    ranked = sorted(frame["name"], key=lambda n: -counts.get((state, last_name(n)), 0))
    return ranked[0]


#### Clean ####
raw = pd.read_csv(config.CANDIDATES_RAW)
raw["state"] = raw["state"].map(config.STATE_ABBR)
assert raw["state"].notna().all(), "A state name in the candidate file is not recognised"

# DFL is Minnesota's name for the Democratic party
raw["party"] = raw["party"].replace({"DFL": "Democratic"})
raw["party_group"] = raw["party"].where(
    raw["party"].isin(["Democratic", "Republican", "Independent", "Libertarian"]), "Other")

counts = poll_mentions()
rows, warnings = [], []

for state, group in raw.groupby("state", sort=False):
    race_type = group["race_type"].iloc[0]
    dems = group[group["party"] == "Democratic"]
    reps = group[group["party"] == "Republican"]
    inds = group[group["party"] == "Independent"]

    if len(dems):
        left_name, left_party = pick(dems, state, counts), "Democratic"
        if len(dems) > 1:
            warnings.append(f"{state}: {len(dems)} Democrats listed; used {left_name}")
    elif len(inds):
        left_name, left_party = pick(inds, state, counts), "Independent"
        warnings.append(f"{state}: no Democrat in the file; used independent {left_name}. CHECK")
    else:
        raise ValueError(f"{state}: no Democrat or independent in the candidate file")

    if len(reps) == 0:
        raise ValueError(f"{state}: no Republican in the candidate file")
    rep_name = pick(reps, state, counts)
    if len(reps) > 1:
        warnings.append(f"{state}: {len(reps)} Republicans listed; used {rep_name}")

    over = OVERRIDES.get(state, {})
    left_name = over.get("left_name", left_name)
    left_party = over.get("left_party", left_party)
    rep_name = over.get("rep_name", rep_name)

    # Everyone else is "other". A second Republican is the same side, so it is not listed.
    skip = {last_name(left_name), last_name(rep_name)}
    others = group[~group["name"].map(last_name).isin(skip) & (group["party"] != "Republican")]
    others = others[~((others["party"] == "Democratic") & (left_party == "Democratic"))]

    rows.append({
        "state": state, "state_name": config.STATE_NAME[state], "race_type": race_type,
        "left_name": left_name, "left_party": left_party, "rep_name": rep_name,
        "other_names": "; ".join(f"{n} ({p})" for n, p in zip(others["name"], others["party"])),
        "n_others": len(others),
    })

nominees = pd.DataFrame(rows)

# Long file with the same columns as the simulated candidates file
long_rows = []
for _, r in nominees.iterrows():
    long_rows.append({"state": r["state"], "race_type": r["race_type"], "name": r["left_name"], "party": r["left_party"]})
    long_rows.append({"state": r["state"], "race_type": r["race_type"], "name": r["rep_name"], "party": "Republican"})
candidates_clean = pd.DataFrame(long_rows)


#### Checks against the polls (names polls show that our file does not) ####
if counts:
    polls = pd.read_csv(config.POLLS_FILE, encoding="utf-8-sig")
    polls = polls[(polls["end_date"] >= "2026-09-01") & polls["party"].isin(["DEM", "REP"])]
    known = {(r["state"], last_name(n)) for _, r in nominees.iterrows()
             for n in (r["left_name"], r["rep_name"])}
    seen = polls[~polls["response_name"].str.contains("Generic", na=False)]
    seen = seen.groupby(["state", "response_name"]).size().reset_index(name="rows")
    seen = seen[seen["rows"] >= 3]
    for _, r in seen.iterrows():
        if (r["state"], last_name(r["response_name"])) not in known:
            warnings.append(f"{r['state']}: recent polls name {r['response_name']} "
                            f"({r['rows']} rows), who is not our nominee. Check the candidate file")


#### Save ####
config.SHARED_ANALYSIS.mkdir(parents=True, exist_ok=True)
nominees.to_csv(config.NOMINEES_FILE, index=False)
candidates_clean.to_csv(config.CANDIDATES_CLEAN, index=False)

assert nominees["state"].nunique() == 35, "Expected 35 races"
print(f"Saved {len(nominees)} races to {config.NOMINEES_FILE}\n")
print(nominees[["state", "race_type", "left_name", "left_party", "rep_name", "n_others"]].to_string(index=False))
print("\nWARNINGS (check each on Ballotpedia, then edit OVERRIDES at the top of this script):")
for w in warnings:
    print("  -", w)
