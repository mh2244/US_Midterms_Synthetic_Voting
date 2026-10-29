#### Preamble ####
# Purpose: Describes who is in the persona sample: counts and shares for each persona variable,
#          overall and by state, next to what the weighted CES pool would give, plus how many
#          personas are repeated draws of the same respondent.
# Author: (group)
# Date: 25 October 2026
# License: MIT
# Pre-requisites: Run scripts/06-clean_candidates.py and scripts/07-build_personas.py first
#   (personas.parquet or personas.csv, and the CES file). Does not call any model or need a key.
# Any other information needed? Run from the project root:
#   python scripts/14-sample_composition.py
#   python scripts/14-sample_composition.py --personas path/to/personas.csv   <- another copy
# Outputs (in data/03-results/composition/):
#   composition_overall.csv    one row per variable and category
#   composition_by_state.csv   long format: state, variable, category, n, pct
#   repeats_by_state.csv       personas, distinct respondents, repeated draws, pool size, Kish n_eff
#   composition_overall.md     the figure and the tables (as a "Table view"), included in index.qmd
#   figures/composition.png    sample against expected shares, one panel per variable (after the
#                              layout of Figure 1 in Murray and Alexander 2026)
#
# HOW to read the "expected" column: each state contributes 100 personas, so the expected overall
# share is the average of the states' weighted CES shares, each state counting equally. A gap
# between the sample and the expected share is sampling noise (and, in Wyoming and Alaska,
# repeated draws), not an error in the weights. The weights were used once, to draw the personas.
# LIMIT: the shares count personas, not people. A respondent drawn twice is two personas.

#### Workspace setup ####
import argparse

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

ap = argparse.ArgumentParser()
ap.add_argument("--personas", default=None, help="path to a personas .csv or .parquet (default: from config)")
args = ap.parse_args()

OUT = config.RESULTS / "composition"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = {
    "gender": "Gender", "race_5": "Race and ethnicity", "age_group": "Age group",
    "education": "Education", "family_income": "Household income", "party_id_7": "Party identification",
    "ideology": "Ideology", "pres_2024": "2024 presidential vote (not in the prompt)",
}
ORDER = {
    "gender": ["Man", "Woman", "Non-binary", "Other"],
    "race_5": ["White", "Black", "Hispanic", "Asian", "Other"],
    "age_group": ["18-29", "30-44", "45-64", "65+"],
    "education": ["No HS", "HS grad", "Some college", "2-year", "4-year", "Post-grad"],
    "party_id_7": ["Strong Dem", "Weak Dem", "Lean Dem", "Independent", "Lean Rep", "Weak Rep",
                   "Strong Rep", "Not sure"],
    "ideology": ["Very liberal", "Liberal", "Moderate", "Conservative", "Very conservative", "Not sure"],
    "family_income": ["Less than $10,000", "$10,000 - $19,999", "$20,000 - $29,999", "$30,000 - $39,999",
                      "$40,000 - $49,999", "$50,000 - $59,999", "$60,000 - $69,999", "$70,000 - $79,999",
                      "$80,000 - $99,999", "$100,000 - $119,999", "$120,000 - $149,999",
                      "$150,000 - $199,999", "$200,000 - $249,999", "$250,000 - $349,999",
                      "$350,000 - $499,999", "$500,000 or more", "Prefer not to say"],
    "pres_2024": ["Harris", "Trump", "Third party", "Did not vote for president", "No answer"],
}
FIGURE = Path("figures") / "composition.png"
# The figure groups the 17 income categories into 7 bands so the panel stays readable
INCOME_BAND = {
    "Less than $10,000": "Under $30K", "$10,000 - $19,999": "Under $30K", "$20,000 - $29,999": "Under $30K",
    "$30,000 - $39,999": "$30K to $50K", "$40,000 - $49,999": "$30K to $50K",
    "$50,000 - $59,999": "$50K to $70K", "$60,000 - $69,999": "$50K to $70K",
    "$70,000 - $79,999": "$70K to $100K", "$80,000 - $99,999": "$70K to $100K",
    "$100,000 - $119,999": "$100K to $150K", "$120,000 - $149,999": "$100K to $150K",
    "$150,000 - $199,999": "$150K or more", "$200,000 - $249,999": "$150K or more",
    "$250,000 - $349,999": "$150K or more", "$350,000 - $499,999": "$150K or more",
    "$500,000 or more": "$150K or more", "Prefer not to say": "Prefer not to say"}


def pres_group(x):
    """2024 presidential vote as reported in the CES, in five groups."""
    if pd.isna(x):
        return "No answer"
    if x in ("Harris", "Trump"):
        return x
    return "Did not vote for president" if x == "Did not vote for President" else "Third party"

VARS = list(LABELS)

#### Load the sample ####
if args.personas:
    path = args.personas
    personas = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
else:
    personas = config.read_table(config.PERSONAS_FILE)
nominees = pd.read_csv(config.NOMINEES_FILE)
states = list(nominees["state"])
assert personas["state"].nunique() == len(states), "Personas do not cover every race state"
per_state = personas.groupby("state").size()
print(f"{len(personas):,} personas in {personas['state'].nunique()} states "
      f"({per_state.min()} to {per_state.max()} per state)")

#### The pool, built the same way as in script 07 ####
ces = pd.read_csv(config.CES_FILE, low_memory=False)
pool = ces[ces["state"].isin(states) & ces["vvweight_post"].notna() & (ces["vvweight_post"] > 0)].copy()
pool = pool[pool["validated_voter"].astype(str).str.lower() == "true"].copy()   # as in 07: validated voters
pool["party_id_7"] = pool["party_id_7"].fillna("Not sure")
pool["ideology"] = pool["ideology"].fillna("Not sure")
pool["family_income"] = pool["family_income"].fillna("Prefer not to say")
pool = pool.dropna(subset=["age_group", "education", "gender", "race_5"])
print(f"Pool: {len(pool):,} validated 2024 voters with vvweight_post in these states")
# 2024 vote: shown for comparison only; the models never see it
pool["pres_2024"] = pool["pres_vote_2024"].map(pres_group)
personas["pres_2024"] = personas["caseid"].map(
    ces.drop_duplicates("caseid").set_index("caseid")["pres_vote_2024"]).map(pres_group)


def categories(var):
    """Listed order first, then any category found in the data that the list does not name."""
    seen = set(personas[var].dropna()) | set(pool[var].dropna())
    return [c for c in ORDER[var] if c in seen] + sorted(seen - set(ORDER[var]))


#### Overall table ####
rows = []
for var in VARS:
    cats = categories(var)
    n = personas[var].value_counts().reindex(cats).fillna(0).astype(int)
    pct = n / n.sum() * 100
    # expected: weighted share inside each state, then the plain average over states
    exp = pd.concat(
        [g.groupby(var)["vvweight_post"].sum().reindex(cats).fillna(0) / g["vvweight_post"].sum()
         for _, g in pool.groupby("state")], axis=1).mean(axis=1) * 100
    unw = pool[var].value_counts(normalize=True).reindex(cats).fillna(0) * 100
    for c in cats:
        rows.append({"variable": LABELS[var], "category": c, "n_sample": int(n[c]),
                     "pct_sample": round(pct[c], 1), "pct_expected": round(exp[c], 1),
                     "pct_pool_unweighted": round(unw[c], 1),
                     "gap_points": round(pct[c] - exp[c], 1)})
overall = pd.DataFrame(rows)
overall.to_csv(OUT / "composition_overall.csv", index=False)

#### By state (long format) ####
long = []
for var in VARS:
    cats = categories(var)
    counts = personas.groupby(["state", var]).size().unstack(fill_value=0).reindex(columns=cats, fill_value=0)
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    for state in counts.index:
        for c in cats:
            long.append({"state": state, "variable": LABELS[var], "category": c,
                         "n": int(counts.loc[state, c]), "pct": round(pct.loc[state, c], 1)})
pd.DataFrame(long).to_csv(OUT / "composition_by_state.csv", index=False)

#### Repeats ####
rep = personas.groupby("state").agg(
    personas=("persona_id", "size"),
    distinct_respondents=("caseid", "nunique"),
    repeated_draws=("repeat_draw", "sum"),
    pool_size=("n_pool", "first"),
    kish_n_eff=("n_eff", "first"),
).reset_index()
rep["kish_n_eff"] = rep["kish_n_eff"].round(1)
rep["pct_repeated"] = (rep["repeated_draws"] / rep["personas"] * 100).round(1)
rep = rep.sort_values("pct_repeated", ascending=False)
rep.to_csv(OUT / "repeats_by_state.csv", index=False)

#### Markdown for index.qmd ####
# index.qmd includes this file ({{< include >}}), so the tables on the page always match the data.
per_state_text = str(per_state.min()) if per_state.min() == per_state.max() else f"{per_state.min()} to {per_state.max()}"
max_gap = overall["gap_points"].abs().max()
md = [f"The sample has {len(personas):,} personas, {per_state_text} in each of the "
      f"{personas['state'].nunique()} states. The expected share is the weighted CES share among "
      "validated 2024 voters in each state, averaged over states, because every state contributes "
      f"the same number of personas. On every category, the sample is within {max_gap:.1f} "
      "percentage points of the expected share.", "",
      f"![Composition of the {len(personas):,} personas (black) against the expected shares from "
      "the weighted CES pool (grey). Household income is grouped into seven bands here; the table "
      f"view has all 17 categories.]({FIGURE.as_posix()}){{fig-alt=\"Paired horizontal bars for gender, "
      "race and ethnicity, age, education, party identification, ideology, household income and "
      "2024 presidential vote, comparing the persona sample with the expected shares.\"}", "",
      "<details>", "<summary>Table view</summary>", ""]


def table(var):
    out = [f"**{LABELS[var]}**", "", "| Category | Personas | Sample (%) | Expected (%) |", "|---|---:|---:|---:|"]
    for _, r in overall[overall["variable"] == LABELS[var]].iterrows():
        out.append(f"| {r['category']} | {r['n_sample']:,} | {r['pct_sample']:.1f} | {r['pct_expected']:.1f} |")
    return out + ["", "::: {.chart-note}",
                  "Sample is the share of drawn personas; expected is the weighted CES share, averaged equally across states. Small differences are expected from random sampling.", ":::", ""]


for left, right in [("gender", "race_5"), ("age_group", "education"), ("party_id_7", "ideology")]:
    md += ["::: {.columns}", "", '::: {.column width="48%"}'] + table(left) + [":::", "",
           '::: {.column width="4%"}', ":::", "", '::: {.column width="48%"}'] + table(right) + [":::", "", ":::", ""]
for left, right in [("family_income", "pres_2024")]:
    md += ["::: {.columns}", "", '::: {.column width="48%"}'] + table(left) + [":::", "",
           '::: {.column width="4%"}', ":::", "", '::: {.column width="48%"}'] + table(right) + [":::", "", ":::", ""]
md += ["</details>", ""]
md += ["**Repeated draws**", "",
       f"Of the {len(personas):,} personas, {int(rep['repeated_draws'].sum()):,} "
       f"({rep['repeated_draws'].sum() / len(personas) * 100:.1f}%) repeat a respondent already drawn "
       "in the same state. The five states with the most repeats are shown below.", "",
       "| State | Pool | Repeated draws | Repeated (%) |", "|---|---:|---:|---:|"]
for _, r in rep.head(5).iterrows():
    md.append(f"| {r['state']} | {int(r['pool_size']):,} | {int(r['repeated_draws'])} | {r['pct_repeated']:.1f} |")
md += ["", "::: {.chart-note}",
       "Pool is the number of eligible CES respondents in the state. Repeated draws count selections after a respondent's first appearance, not duplicate API calls. Sampling with replacement allows one respondent to appear as more than one persona.", ":::", ""]
(OUT / "composition_overall.md").write_text("\n".join(md).rstrip() + "\n")

#### Figure ####
# Two columns of panels; in each panel the black bar is the sample, the grey bar the expected share.
INK, GREY = config.INK, "#b3b3b3"
config.use_site_style()            # shared graph theme: scripts/site.mplstyle
fig_data = overall.copy()
inc = fig_data["variable"] == LABELS["family_income"]
fig_data.loc[inc, "category"] = fig_data.loc[inc, "category"].map(INCOME_BAND)
fig_data = fig_data.groupby(["variable", "category"], sort=False)[["pct_sample", "pct_expected"]].sum().reset_index()
order = ["gender", "race_5", "age_group", "education", "party_id_7", "ideology", "family_income", "pres_2024"]
rows_n = [max(len(fig_data[fig_data["variable"] == LABELS[order[i]]]),
              len(fig_data[fig_data["variable"] == LABELS[order[i + 1]]])) for i in range(0, len(order), 2)]
fig, axes = plt.subplots(len(rows_n), 2, figsize=(8, 0.25 * sum(rows_n) + 2.2),
                         gridspec_kw={"height_ratios": rows_n, "hspace": 0.55, "wspace": 0.75})
xmax = np.ceil(fig_data[["pct_sample", "pct_expected"]].to_numpy().max() / 10) * 10
for i, (ax, var) in enumerate(zip(axes.flat, order)):
    sub = fig_data[fig_data["variable"] == LABELS[var]]
    rowmax = rows_n[i // 2]
    y = np.arange(len(sub))[::-1] + (rowmax - len(sub))     # top-aligned, same bar size in a row
    ax.barh(y + 0.19, sub["pct_sample"], height=0.36, color=INK)
    ax.barh(y - 0.19, sub["pct_expected"], height=0.36, color=GREY)
    ax.set_yticks(y, [c.replace("$", r"\$") for c in sub["category"]])
    ax.set_ylim(-0.6, rowmax - 0.4)
    ax.set_xlim(0, xmax)
    ax.set_xticks(np.arange(0, xmax + 0.1, 25), [f"{v:.0f}%" for v in np.arange(0, xmax + 0.1, 25)])
    ax.set_title(LABELS[var].replace(" (not in the prompt)", "\n(not in the prompt)"), loc="left",
                 fontsize=10, fontweight="bold")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=INK), plt.Rectangle((0, 0), 1, 1, color=GREY)]
fig.legend(handles, [f"The {len(personas):,} personas", "Expected (weighted CES, validated 2024 voters)"],
           loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.07)
FIGURE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE, dpi=200)
plt.close(fig)
print(f"Saved {FIGURE}")

#### Checks ####
assert overall.groupby("variable")["n_sample"].sum().eq(len(personas)).all(), "A variable does not add up"
assert (rep["personas"] == rep["distinct_respondents"] + rep["repeated_draws"]).all(), \
    "personas should equal distinct respondents plus repeated draws"

print(f"\nSaved tables to {OUT}/")
print("\nLargest gaps between sample and expected shares (points):")
print(overall.reindex(overall["gap_points"].abs().sort_values(ascending=False).index).head(8)
      [["variable", "category", "pct_sample", "pct_expected", "gap_points"]].to_string(index=False))
print("\nMost repeated draws:")
print(rep.head(5).to_string(index=False))
