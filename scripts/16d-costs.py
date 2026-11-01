#### Preamble ####
# Purpose: Cost and timing of the model queries, for the "API costs and time" section of the
#          website: answers, tokens and cost in US dollars by model for the latest news run, the
#          cost of every run, and the time from the latest run's first call to its last, from
#          the token counts that script 09 saves with every answer (all runs are in costs.csv).
# Author: Kexin (Coco) Liu
# Date: 27 October 2026
# License: MIT
# Pre-requisites: Script 09 has run for at least one run.
# Any other information needed? Run from the project root:  python scripts/16d-costs.py
# Outputs: data/03-results/comparison/costs.csv and costs.md
#
# HOW the cost is computed: input and output tokens times the OpenRouter list prices in
#   config.PRICES (US dollars per million tokens, checked 25 October 2026). DeepSeek is cheaper
#   off-peak; the list price is used, so its cost is an upper bound. Failed calls that returned no
#   token counts cost nothing here.
# NOT included: the SSR embeddings (Gemini Embedding 2) and the news screening (GPT-4.1 Nano), whose
#   token counts are not saved; take those from the OpenRouter activity page.
# NOTE on time: a run that is stopped and resumed spans the gap between attempts.

#### Workspace setup ####
import pandas as pd

import config

RUNS = {"2026-10-27": "27 October", "2026-10-29": "29 October", "2026-10-31": "31 October",
        "nonews": "No news (baseline)"}
OUT = config.RESULTS / "comparison"

#### Tokens and cost by run and model ####
rows = []
for run, name in RUNS.items():
    path = config.SHARED_ANALYSIS / f"run_{run}" / "responses.parquet"
    if not path.exists():
        continue
    r = pd.read_parquet(path)
    r["time"] = pd.to_datetime(r["queried_at_utc"], utc=True, errors="coerce")
    for model, g in r.groupby("model"):
        price_in, price_out = config.PRICES[model]
        tin, tout = g["input_tokens"].fillna(0).sum(), g["output_tokens"].fillna(0).sum()
        ok = (g["error"].fillna("") == "") & (g["response_text"].fillna("").str.strip() != "")
        rows.append({"run": run, "run_label": name, "model": model, "calls": len(g), "answered": int(ok.sum()),
                     "input_tokens": int(tin), "output_tokens": int(tout),
                     "cost_usd": tin / 1e6 * price_in + tout / 1e6 * price_out,
                     "first_call": g["time"].min(), "last_call": g["time"].max()})
costs = pd.DataFrame(rows)
assert len(costs), "No run with model answers yet"
assert (costs["cost_usd"] >= 0).all() and costs["cost_usd"].notna().all()
OUT.mkdir(parents=True, exist_ok=True)
costs.round({"cost_usd": 4}).to_csv(OUT / "costs.csv", index=False)

#### Table: the latest news run, one row per model ####
dated = [r for r in RUNS if r != "nonews" and r in set(costs["run"])]
latest = dated[-1] if dated else costs["run"].iloc[-1]
g = costs[costs["run"] == latest]
md = [f"**{RUNS[latest]} run**", "",
      "| Model | Answers | Input tokens | Output tokens | Cost (US$) |", "|---|---:|---:|---:|---:|"]
for r in g.itertuples():
    md.append(f"| {config.MODEL_LABELS.get(r.model, r.model)} | {r.answered:,} | {r.input_tokens:,} | "
              f"{r.output_tokens:,} | {r.cost_usd:,.2f} |")
md.append(f"| **All models** | **{g['answered'].sum():,}** | **{g['input_tokens'].sum():,}** | "
          f"**{g['output_tokens'].sum():,}** | **{g['cost_usd'].sum():,.2f}** |")
# One sentence for every run, so earlier runs stay on the page without their own rows
totals = costs.groupby("run", sort=False).agg(cost=("cost_usd", "sum"), first=("first_call", "min"),
                                              last=("last_call", "max"))
parts = [f"{RUNS[r].replace(' (baseline)', '').replace('No news', 'the run without news')} US${t.cost:,.2f}"
         for r, t in totals.iterrows()]
hours = (totals.loc[latest, "last"] - totals.loc[latest, "first"]).total_seconds() / 3600
md += ["", f"Cost of each run: {', '.join(parts[:-1])} and {parts[-1]}. The {RUNS[latest]} run took "
           f"{hours:.1f} hours from its first call to its last."]
(OUT / "costs.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
