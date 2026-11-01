#### Preamble ####
# Purpose: Crosstabs of the synthetic vote: for each demographic group, the share of the
#          two-candidate vote that goes to the Democratic (or independent) candidate, next to how
#          the SAME personas voted for president in 2024 (Harris share of Harris + Trump, CES).
# Author: Kexin (Coco) Liu
# Date: 26 October 2026
# License: MIT
# Pre-requisites: Run scripts 07 (personas) and 10 (scores) for this RUN. Needs the CES file.
#   Does not call any model or need a key.
# Any other information needed? Run from the project root:
#   python scripts/16-make_crosstabs.py              <- RUN from config / the RUN variable
#   RUN=nonews python scripts/16-make_crosstabs.py
# Outputs (in data/03-results/run_<RUN>/):
#   crosstabs.csv          one row per model, variable and category (plus the all-model mean)
#   crosstabs.md           one table for index.qmd ({{< include >}} it)
#
# HOW a cell is built:
#   1. Average each persona's p_left and p_rep over repeats (Design B).
#   2. Weight each persona by its state's share of the validated-voter weight (sum of
#      vvweight_post in the state / 100 personas), so that pooled groups reflect the electorate
#      of the 35 states rather than counting every state equally.
#   3. Cell share = sum(w * p_left) / sum(w * (p_left + p_rep)), for each model; the table shows
#      the plain mean over models (each model counts once) and the range across models.
#   4. "Leaning left" is the weighted share of personas whose p_left is larger than p_rep. SSR
#      rarely gives a probability near 0 or 1 (an answer that clearly backs one candidate still
#      scores about 0.85 to 0.9), so the share in step 3 is pulled toward 50 per cent in groups that
#      vote one way; the leaning-left share shows which side the answers actually took.
#   5. The 2024 column uses the same personas and the same weights, from pres_vote_2024 in the
#      CES (Harris or Trump only). The models never see the 2024 vote.
# WHY no poll correction here: polls do not report these groups in every state, so the crosstabs
#   describe the synthetic voters as they answered, before the combination in script 11.
# LIMIT: a persona drawn twice counts twice; cells under 30 personas are marked and are noisy.

#### Workspace setup ####
import numpy as np
import pandas as pd

import config

LABELS = {
    "gender": "Gender", "race_5": "Race and ethnicity", "age_group": "Age group",
    "education": "Education", "party_id_7": "Party identification", "ideology": "Ideology",
    "income_band": "Household income",
}
ORDER = {
    "gender": ["Man", "Woman", "Non-binary", "Other"],
    "race_5": ["White", "Black", "Hispanic", "Asian", "Other"],
    "age_group": ["18-29", "30-44", "45-64", "65+"],
    "education": ["No HS", "HS grad", "Some college", "2-year", "4-year", "Post-grad"],
    "party_id_7": ["Strong Dem", "Weak Dem", "Lean Dem", "Independent", "Lean Rep", "Weak Rep",
                   "Strong Rep", "Not sure"],
    "ideology": ["Very liberal", "Liberal", "Moderate", "Conservative", "Very conservative", "Not sure"],
    "income_band": ["Under $50K", "$50K to $100K", "$100K to $200K", "$200K or more", "Prefer not to say"],
}


def income_band(x):
    """Collapse the 17 CES income categories into 5 bands, so cells are not tiny."""
    if pd.isna(x) or x == "Prefer not to say":
        return "Prefer not to say"
    low = float(str(x).replace("Less than ", "0 - ").split(" - ")[0].replace("$", "").replace(",", "")
                .replace(" or more", ""))
    if low < 50_000:
        return "Under $50K"
    if low < 100_000:
        return "$50K to $100K"
    if low < 200_000:
        return "$100K to $200K"
    return "$200K or more"


#### Data ####
print(f"Run: {config.RUN}")
personas = config.read_table(config.PERSONAS_FILE)
personas["income_band"] = personas["family_income"].map(income_band)
scores = config.read_table(config.SCORES_FILE)
scores = scores[(scores["design"] == "B") & scores["p_left"].notna()]
per = scores.groupby(["persona_id", "model"])[["p_left", "p_rep"]].mean().reset_index()

# State weights from the same validated-voter pool as script 07
ces = pd.read_csv(config.CES_FILE, low_memory=False,
                  usecols=["caseid", "state", "vvweight_post", "validated_voter", "pres_vote_2024"])
states = personas["state"].unique()
pool = ces[ces["state"].isin(states) & (ces["vvweight_post"] > 0)
           & (ces["validated_voter"].astype(str).str.lower() == "true")]
state_w = pool.groupby("state")["vvweight_post"].sum()
personas["w"] = personas["state"].map(state_w) / personas["state"].map(personas.groupby("state").size())
assert personas["w"].notna().all(), "A persona's state has no weight"
pres = ces.drop_duplicates("caseid").set_index("caseid")["pres_vote_2024"]
personas["harris"] = personas["caseid"].map(pres).map({"Harris": 1.0, "Trump": 0.0})

df = per.merge(personas[["persona_id", "w"] + list(LABELS)], on="persona_id", how="left")
assert df["w"].notna().all(), "Scores contain persona_id values that are not in the personas file"


#### Cells ####
def cell_share(g):
    return (g["w"] * g["p_left"]).sum() / (g["w"] * (g["p_left"] + g["p_rep"])).sum()


def lean_left(g):
    return (g["w"] * (g["p_left"] > g["p_rep"])).sum() / g["w"].sum()


rows = []
groups = [("All", "All personas", None)] + [(v, LABELS[v], v) for v in LABELS]
for var, label, col in groups:
    cats = ["All personas"] if col is None else \
        [c for c in ORDER[col] if c in set(personas[col])] + sorted(set(personas[col].dropna()) - set(ORDER[col]))
    for cat in cats:
        sel_p = personas if col is None else personas[personas[col] == cat]
        sel_d = df if col is None else df[df[col] == cat]
        h = sel_p.dropna(subset=["harris"])
        harris = (h["w"] * h["harris"]).sum() / h["w"].sum() if len(h) else np.nan
        for model, g in sel_d.groupby("model"):
            rows.append({"variable": label, "category": cat, "model": model, "n_personas": len(sel_p),
                         "n_scored": g["persona_id"].nunique(), "share_2026": cell_share(g),
                         "lean_left": lean_left(g),
                         "harris_2024": harris})
tab = pd.DataFrame(rows)
mean = (tab.groupby(["variable", "category"], sort=False)
        .agg(n_personas=("n_personas", "first"), share_2026=("share_2026", "mean"),
             low=("share_2026", "min"), high=("share_2026", "max"), lean_left=("lean_left", "mean"),
             harris_2024=("harris_2024", "first"))
        .reset_index())
# Difference = leaning left minus the 2024 Harris share of the same personas. It uses the
# leaning-left share, not the SSR share, so the pull toward 50 per cent does not show up as change.
mean["change"] = mean["lean_left"] - mean["harris_2024"]

#### Checks ####
assert tab["share_2026"].between(0, 1).all(), "A cell share is outside 0 to 1"
assert tab["harris_2024"].dropna().between(0, 1).all()
assert tab["lean_left"].between(0, 1).all()
assert tab.groupby(["variable", "category"])["model"].nunique().eq(tab["model"].nunique()).all(), \
    "A model is missing from a cell"
for label in LABELS.values():   # every variable's categories add up to all personas
    assert mean.loc[mean["variable"] == label, "n_personas"].sum() == len(personas), label

#### Save ####
config.RUN_RESULTS.mkdir(parents=True, exist_ok=True)
out = pd.concat([tab, mean.assign(model="mean of models")], ignore_index=True)
out.round(4).to_csv(config.RUN_RESULTS / "crosstabs.csv", index=False)

def pct(x):
    return "" if pd.isna(x) else f"{x * 100:.0f}"


def signed(x):
    """Change in points, with a proper minus sign and no '-0'."""
    if pd.isna(x):
        return ""
    v = int(round(x * 100))
    return "0" if v == 0 else f"+{v}" if v > 0 else f"\u2212{-v}"


# Two narrow tables side by side, written as HTML so each variable is its own block (tbody) with
# a shaded header row; the styling is in styles.css (.crosstab). Each row: group, persona count,
# share of synthetic voters leaning left, share of the same personas who voted Harris in 2024, and
# the difference. The SSR share (share_2026) and its range across models stay in crosstabs.csv but
# are not shown: the page compares people with people (leaning left against voted Harris).
import html as html_lib

HEAD = ("<thead><tr><th>Group</th><th>Personas</th><th>Synthetic voters leaning left (%)</th>"
        "<th>Voted Harris in 2024 (%)</th><th>Difference (points)</th></tr></thead>")


def block(label):
    sub = mean[mean["variable"] == label]
    rows = [] if label == "All personas" else [f'<tr class="ct-head"><th colspan="5">{html_lib.escape(label)}</th></tr>']
    for _, r in sub.iterrows():
        cls = ' class="ct-all"' if label == "All personas" else ""
        rows.append(f"<tr{cls}><td>{html_lib.escape(r['category'])}</td><td>{r['n_personas']:,}</td>"
                    f"<td>{pct(r['lean_left'])}</td><td>{pct(r['harris_2024'])}</td>"
                    f"<td>{signed(r['change'])}</td></tr>")
    return "<tbody>" + "".join(rows) + "</tbody>"


left = ["All personas", "Gender", "Race and ethnicity", "Age group", "Education"]
right = ["Party identification", "Ideology", "Household income"]
assert set(left + right) == {"All personas", *LABELS.values()}, "A crosstab variable is not placed"
tables = "".join(f'<div class="crosstab-col"><table class="crosstab">{HEAD}{"".join(block(v) for v in side)}'
                 "</table></div>" for side in (left, right))
md = ["```{=html}", f'<div class="crosstab-pair">{tables}</div>', "```"]
(config.RUN_RESULTS / "crosstabs.md").write_text("\n".join(md) + "\n")

print(f"Saved {config.RUN_RESULTS / 'crosstabs.csv'} and crosstabs.md")
print(mean.assign(**{c: mean[c].round(3) for c in ["share_2026", "low", "high", "lean_left", "harris_2024", "change"]})
      .to_string(index=False))
