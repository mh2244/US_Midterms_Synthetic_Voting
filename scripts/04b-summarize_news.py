#### Preamble ####
# Purpose: Samples three items from each news source group for every persona and
#          concatenates their existing summaries into persona-specific prompt context.
# Author: Jingxuan Feng
# Date: 26 October 2026
# Contact: amelia.f.feng@outlook.com
# License: MIT
# Pre-requisites: Run scripts 04 and 07; install requirements.txt.
#                 This script makes no API calls and needs no API key.
# Run: python scripts/04b-summarize_news.py
# Preview selections without writing output: add --dry-run.
# Output: data/news/persona_news_summaries.csv (includes all selected sources).
# newspapers is a source group, NOT a claim of political neutrality.

import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config

GROUPS = ("left_leaning", "newspapers", "right_leaning")
ARTICLE_FIELDS = ["url", "title", "outlet", "diet", "published_at_utc", "prompt_paragraph"]
SUMMARY_METHOD = "concatenate_v1"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def persona_fingerprint(row):
    return fingerprint({k: str(row[k]) for k in ("persona_id", "state", "persona_text")})


def load_news(path):
    news = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    required = set(ARTICLE_FIELDS) | {"llm_status", "national", "election_relevant"}
    missing = required - set(news.columns)
    if missing:
        raise ValueError(f"News file missing columns: {', '.join(sorted(missing))}")
    keep = news["llm_status"].eq("keep")
    for flag in ("national", "election_relevant"):
        keep &= news[flag].astype(str).str.lower().eq("true")
    news = news.loc[keep, ARTICLE_FIELDS].copy()
    for column in ARTICLE_FIELDS:
        news[column] = news[column].astype(str).str.strip()
    news = news[news["url"].ne("") & news["prompt_paragraph"].ne("")]
    # Stable order and unique URLs make sampling independent of CSV row ordering.
    news = news.sort_values(ARTICLE_FIELDS).drop_duplicates("url").reset_index(drop=True)
    counts = news["diet"].value_counts()
    for group in GROUPS:
        if counts.get(group, 0) < 3:
            raise ValueError(f"Need at least 3 unique articles in {group}; found {counts.get(group, 0)}. "
                             "Collect more news; no replacement or silent rebalancing is used.")
    news.attrs["file_hash"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return news


def make_plan(personas, news, run_id, seed):
    if personas.empty or personas["persona_id"].duplicated().any():
        raise ValueError("Personas must be nonempty with unique persona_id values")
    if personas[["persona_id", "state", "persona_text"]].isna().any().any():
        raise ValueError("Personas contain missing identifiers or descriptions")
    pool_hash = fingerprint(news.to_dict("records"))
    plan = []
    for person in personas.to_dict("records"):
        person_hash = persona_fingerprint(person)
        rng = np.random.default_rng(int(fingerprint([seed, run_id, person_hash, pool_hash])[:16], 16))
        selected = []
        for group in GROUPS:
            records = news[news["diet"] == group].to_dict("records")
            selected.extend(records[int(i)] for i in rng.choice(len(records), 3, replace=False))
        # Mix presentation order rather than always presenting the left group first.
        selected = [selected[int(i)] for i in rng.permutation(9)]
        articles = [{"id": i, **article} for i, article in enumerate(selected, 1)]
        request_hash = fingerprint([SUMMARY_METHOD, articles])
        plan.append({"persona_id": person["persona_id"], "state": person["state"],
                     "run_id": run_id, "seed": seed, "persona_hash": person_hash,
                     "news_pool_hash": pool_hash, "news_file_hash": news.attrs["file_hash"],
                     "request_hash": request_hash,
                     "articles_json": json.dumps(articles, ensure_ascii=False),
                     "summary_method": SUMMARY_METHOD, "status": "ok",
                     "summary": concatenate_summaries(articles),
                     "generated_at_utc": datetime.now(timezone.utc).isoformat(), "error": ""})
    return plan


def concatenate_summaries(articles):
    """Preserve all nine existing paragraphs verbatim, separated and numbered."""
    if len(articles) != 9 or len({a["url"] for a in articles}) != 9:
        raise ValueError("Expected nine articles with distinct URLs")
    if any(not str(a["prompt_paragraph"]).strip() for a in articles):
        raise ValueError("Every selected article must have an existing summary")
    return "\n\n".join(f"[{i}] {article['prompt_paragraph']}"
                        for i, article in enumerate(articles, 1))


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Select 3/3/3 news items per persona and concatenate their existing summaries; no API calls.")
    parser.add_argument("--run", default=os.environ.get("RUN") or datetime.now(ZoneInfo("America/Toronto")).date().isoformat())
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--out", type=Path, default=config.NEWS_SUMMARIES_FILE)
    parser.add_argument("--dry-run", action="store_true", help="Preview selections without writing output")
    args = parser.parse_args()
    if args.run == "nonews":
        parser.error("The nonews baseline does not require news summaries")
    try:
        news = load_news(config.NEWS_FILE)
        rows = make_plan(config.read_table(config.PERSONAS_FILE), news, args.run, args.seed)
        print(f"Personas: {len(rows)}; nine existing summaries per persona; no API calls.")
        print("Each persona: 3 left_leaning + 3 newspapers + 3 right_leaning; newspapers does not mean neutral.")
        if args.dry_run:
            print("Preview for the first persona:\n" + rows[0]["summary"])
            return 0
        save(args.out, rows)
        print(f"Saved {len(rows)} concatenated news contexts to {args.out}")
        return 0
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
