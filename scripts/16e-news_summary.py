#### Preamble ####
# Purpose: The News summaries section of the website: for each news run, how many articles were
#          kept from each outlet and source group, what they were about, and the full news block
#          that one persona received, as an example of what the models read.
# Author: Kexin (Coco) Liu
# Date: 28 October 2026
# License: MIT
# Pre-requisites: Scripts 04 and 04b have run for at least one news day.
# Any other information needed? Run from the project root:  python scripts/16e-news_summary.py
# Inputs: data/news/news_context_<RUN>.csv (the articles kept by script 04) and
#   data/news/persona_news_summaries_<RUN>.csv (the nine paragraphs each persona received, 04b).
# Output: data/03-results/comparison/news_summary.md (included by index.qmd).

#### Workspace setup ####
import json
from pathlib import Path

import pandas as pd

import config

NEWS_DAYS = ["2026-10-27", "2026-10-29", "2026-10-31"]
NEWS = Path("data/news")
OUT = config.RESULTS / "comparison"
GROUPS = {"left_leaning": "Left-leaning", "newspapers": "Newspapers", "right_leaning": "Right-leaning"}
TOPICS = {"policy": "Policy", "foreign_policy": "Foreign policy", "law": "Law and courts", "elections": "Elections",
          "voting_rules": "Voting rules", "healthcare": "Health care", "economy": "Economy",
          "immigration": "Immigration", "taxation": "Taxes", "energy": "Energy", "education": "Education",
          "national_security": "National security", "other": "Other"}
EXAMPLE_PERSONA = "FL_12"                # Actual archived block selected for illustration, not at random


def day(run):
    d = pd.Timestamp(run)
    return f"{d.day} {d:%B}"


runs = [r for r in NEWS_DAYS if (NEWS / f"news_context_{r}.csv").exists()]
assert runs, "No news run yet"
ctx = {r: pd.read_csv(NEWS / f"news_context_{r}.csv") for r in runs}

#### Table 1: articles kept by source group and outlet ####
counts = pd.concat({day(r): c.groupby(["diet", "outlet"]).size() for r, c in ctx.items()}, axis=1).fillna(0).astype(int)
md = ["| Source group | Outlet | " + " | ".join(counts.columns) + " |", "|---|---|" + "---:|" * len(counts.columns)]
for group, label in GROUPS.items():
    if group not in counts.index.get_level_values(0):
        continue
    sub = counts.loc[group]
    for i, (outlet, row) in enumerate(sub.iterrows()):
        md.append(f"| {label if i == 0 else ''} | {outlet} | " + " | ".join(f"{v:,}" for v in row) + " |")
    md.append(f"| | *{label}, all outlets* | " + " | ".join(f"*{v:,}*" for v in sub.sum()) + " |")
md.append("| **All** | | " + " | ".join(f"**{v:,}**" for v in counts.sum()) + " |")
table_outlets = "\n".join(md)

#### Table 2: topics of the kept articles (per cent of each run's articles) ####
topics = pd.concat({day(r): c["topic"].value_counts(normalize=True) * 100 for r, c in ctx.items()}, axis=1).fillna(0)
topics = topics.sort_values(topics.columns[-1], ascending=False)
md = ["| Topic | " + " | ".join(topics.columns) + " |", "|---|" + "---:|" * len(topics.columns)]
for t, row in topics.iterrows():
    md.append(f"| {TOPICS.get(t, t)} | " + " | ".join(f"{v:.0f}" for v in row) + " |")
table_topics = "\n".join(md)

#### Example: the news one persona read in the latest run ####
latest = runs[-1]
summaries = pd.read_csv(NEWS / f"persona_news_summaries_{latest}.csv")
ex = summaries[summaries["persona_id"] == EXAMPLE_PERSONA]
ex = ex.iloc[0] if len(ex) else summaries.iloc[0]
articles = json.loads(ex["articles_json"])
example = "\n".join(f"{i}. {a['prompt_paragraph']}" for i, a in enumerate(articles, 1))
n_personas = summaries["persona_id"].nunique()

#### Write ####
pubs = pd.to_datetime(ctx[latest]["published_at_utc"], utc=True)
text = f"""**Articles kept, by outlet.** Number of articles that passed the screening on each news day.

{table_outlets}

::: {{.chart-note}}
Counts refer to retained articles, not the number shown to each persona. Each persona receives three items from each source group; the newspapers group is not a neutrality rating.
:::

**What the kept articles were about.** Share of each news day's kept articles by topic (per cent), as labelled by the screening model.

{table_topics}

::: {{.chart-note}}
Each article has one topic assigned by the screening model. Percentages describe the retained news pool, not the balance of topics in every persona's nine-item selection; rounding may prevent them from adding to 100%.
:::

<details>
<summary>Example: the news block of one persona ({config.STATE_NAME.get(ex['state'], ex['state'])}, {day(latest)})</summary>

Each of the {n_personas:,} personas received nine paragraphs, three from each source group, drawn at random from that day's kept articles (published {pubs.min():%-d %B %H:%M} to {pubs.max():%-d %B %H:%M} UTC). For illustration, we selected the archived block for persona {ex['persona_id']}, reproduced word for word. This example was not randomly selected and does not establish the quality of all news blocks.

{example}

</details>
"""
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "news_summary.md").write_text(text)

#### Checks ####
assert (counts.sum() > 0).all(), "A news run kept no articles"
assert all(len(json.loads(a)) == 9 for a in summaries["articles_json"]), "A persona did not receive nine paragraphs"
print(text)
