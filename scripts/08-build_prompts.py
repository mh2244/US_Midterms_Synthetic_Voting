#### Preamble ####
# Purpose: Builds the prompt for each persona: instructions, optional news summaries, the
#          persona description, and the voting question.
# Author: Maggie, Coco, Amelia
# Date: 22 October 2026
# License: MIT
# Pre-requisites: Run scripts/06-clean_candidates.py and scripts/07-build_personas.py first.
#   Optional: data/news/persona_news_summaries.csv from scripts/04b-summarize_news.py.
# Any other information needed? Run from the project root.
#   --designs B      one free-text answer, scored with SSR in script 10 (the main method)
#   --designs B,C    also a "100 people like you" percentage question (a comparison)
#   --news           include each persona's nine existing news summaries
#
# WHY we save the prompts: the disclosure checklist in Murray & Alexander (2026) asks for the
# full prompts. This file is the exact text the paper can quote.
# WHY the order of the two candidates is randomised: a fixed order (always the Democrat first)
# could push answers toward the first name (prompt wording moves results: Bisbee et al. 2024).
# The column `left_first` records the order used.
# Script 04b samples three items per source group and concatenates the nine existing
# paragraphs without another LLM call. The same context is kept across B/C designs.
# LIMIT: summaries are generated from titles and feed excerpts, not full articles.
# Treat the news run as an experiment, not as the main result.


#### Workspace setup ####
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

import config

ap = argparse.ArgumentParser()
ap.add_argument("--designs", default="B", help="comma list from B,C")
ap.add_argument("--news", action="store_true", help="include nine existing news summaries per persona from script 04b")
args = ap.parse_args()
designs = [d.strip().upper() for d in args.designs.split(",")]
assert set(designs) <= {"B", "C"}, "designs must be B and/or C"

INSTRUCTIONS = (
    "You are taking part in a survey about the November 3, 2026 U.S. elections. "
    "Answer as the person described below would, in the first person. "
    "Do not say that you are an AI model."
)


def load_summaries(personas):
    """Fail rather than silently use unfinished or stale persona summaries."""
    if not config.NEWS_SUMMARIES_FILE.exists():
        raise ValueError("Run scripts/04b-summarize_news.py first to generate persona summaries")
    summaries = pd.read_csv(config.NEWS_SUMMARIES_FILE, encoding="utf-8-sig", dtype=str).fillna("")
    required = {"persona_id", "status", "summary", "persona_hash", "news_file_hash",
                "request_hash", "articles_json", "run_id", "summary_method"}
    if not required <= set(summaries.columns):
        raise ValueError("Summary file is missing required columns; rerun script 04b")
    if summaries["persona_id"].duplicated().any():
        raise ValueError("Summary file contains duplicate persona IDs")
    summaries = summaries.set_index("persona_id")
    source_hash = hashlib.sha256(config.NEWS_FILE.read_bytes()).hexdigest()
    for person in personas.to_dict("records"):
        pid = person["persona_id"]
        if pid not in summaries.index:
            raise ValueError(f"Missing summary for {pid}; rerun script 04b")
        row = summaries.loc[pid]
        person_hash = hashlib.sha256(json.dumps(
            {k: str(person[k]) for k in ("persona_id", "state", "persona_text")},
            sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if row["summary_method"] != "concatenate_v1":
            raise ValueError(f"Old LLM summary for {pid}; rerun script 04b to concatenate the nine items")
        articles = json.loads(row["articles_json"])
        if len(articles) != 9 or len({a["url"] for a in articles}) != 9:
            raise ValueError(f"Expected nine distinct news items for {pid}; rerun script 04b")
        expected = "\n\n".join(f"[{i}] {a['prompt_paragraph']}" for i, a in enumerate(articles, 1))
        if row["summary"] != expected:
            raise ValueError(f"News paragraphs changed for {pid}; rerun script 04b")
        if row["status"] != "ok" or not str(row["summary"]).strip():
            raise ValueError(f"Unfinished summary for {pid}; resume script 04b")
        if row["persona_hash"] != person_hash or row["news_file_hash"] != source_hash:
            raise ValueError(f"Stale persona/news summary for {pid}; rerun script 04b")
        if os.environ.get("RUN") and row["run_id"] != os.environ["RUN"]:
            raise ValueError(f"Summary run id mismatch for {pid}; rerun script 04b")
    return summaries


def news_block(summary):
    return "Here are some of today's news:\n\n" + summary


def candidate_text(race, left_first):
    """'A (Democratic) and B (Republican)' with the order flipped when left_first is False."""
    left = f"{race['left_name']} ({race['left_party']})"
    right = f"{race['rep_name']} (Republican)"
    pair = f"{left} and {right}" if left_first else f"{right} and {left}"
    return pair + (", along with other candidates" if race["n_others"] > 0 else "")


def question_b(race, left_first):
    kind = "special election" if race["race_type"] == "special" else "election"
    return (f"In the 2026 U.S. Senate {kind} in {race['state_name']}, the candidates are "
            f"{candidate_text(race, left_first)}. How do you plan to vote, or do you plan not to vote? "
            "Answer in two or three sentences.")


def question_c(race, left_first):
    """Percentages question. A and B follow the order in which the names are listed."""
    first, second = ((race["left_name"], race["rep_name"]) if left_first
                     else (race["rep_name"], race["left_name"]))
    kind = "special election" if race["race_type"] == "special" else "election"
    return (f"Think of 100 people in {race['state_name']} who share your background and views. "
            f"In the 2026 U.S. Senate {kind}, how many of those 100 would vote for A = {first}, "
            f"how many for B = {second}, how many for another candidate, and how many would not vote? "
            'Reply with only JSON in this form: {"A": 0, "B": 0, "other": 0, "not_voting": 0}. '
            "The four numbers must add up to 100.")


#### Build ####
personas = config.read_table(config.PERSONAS_FILE)
nominees = pd.read_csv(config.NOMINEES_FILE).set_index("state")
rng = np.random.default_rng(config.SEED)
try:
    summaries = load_summaries(personas) if args.news else None
except ValueError as exc:
    ap.error(str(exc))

rows = []
for _, p in personas.iterrows():
    race = nominees.loc[p["state"]].copy()
    left_first = bool(rng.integers(0, 2))    # one random order per persona, kept for all designs
    summary = summaries.loc[p["persona_id"]] if args.news else None
    news_context = news_block(summary["summary"]) if summary is not None else ""
    for design in designs:
        question = question_b(race, left_first) if design == "B" else question_c(race, left_first)
        parts = [INSTRUCTIONS]
        if news_context:
            parts.append(news_context)
        parts += [p["persona_text"], question]
        prompt_text = "\n\n".join(parts)
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()
        rows.append({
            "prompt_id": f"{p['persona_id']}|{design}|news{int(args.news)}|{prompt_hash}",
            "persona_id": p["persona_id"], "state": p["state"], "design": design,
            "news": bool(args.news), "left_first": left_first,
            "news_summary_id": summary["request_hash"] if summary is not None else "",
            "news_articles_json": summary["articles_json"] if summary is not None else "",
            "prompt_text": prompt_text,
        })

prompts = pd.DataFrame(rows)

#### Checks and save ####
assert prompts["prompt_id"].is_unique
assert prompts["news"].eq(args.news).all()
assert prompts["prompt_text"].str.len().gt(100).all()
# News summaries may name politicians, so only the persona text is checked for the 2024 vote
assert not personas["persona_text"].str.contains("Trump|Harris", case=False).any(), "2024 vote leaked into a persona"

config.write_table(prompts, config.PROMPTS_FILE)
print(f"Saved {len(prompts)} prompts ({', '.join(designs)}; news={args.news}) to {config.PROMPTS_FILE}")
print("\nExample prompt:\n" + "-" * 60 + "\n" + prompts["prompt_text"].iloc[0] + "\n" + "-" * 60)
