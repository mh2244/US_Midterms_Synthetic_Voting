#### Preamble ####
# Purpose: Builds the prompt for each persona: instructions, optional news headlines, the
#          persona description, and the voting question.
# Author: Maggie, Coco, Amelia
# Date: 22 October 2026
# License: MIT
# Pre-requisites: Run scripts/06-clean_candidates.py and scripts/07-build_personas.py first.
#   Optional: data/news_google/news_google.csv from scripts/04-collect_news.py.
# Any other information needed? Run from the project root.
#   --designs B      one free-text answer, scored with SSR in script 10 (the main method)
#   --designs B,C    also a "100 people like you" percentage question (a comparison)
#   --news           put the collected headlines before the persona
#
# WHY we save the prompts: the disclosure checklist in Murray & Alexander (2026) asks for the
# full prompts. This file is the exact text the paper can quote.
# WHY the order of the two candidates is randomised: a fixed order (always the Democrat first)
# could push answers toward the first name (prompt wording moves results: Bisbee et al. 2024).
# The column `left_first` records the order used.
# WHY the news goes first: identical leading text can be cached by the provider, which lowers
# cost (Murray & Alexander 2026). Expect only a small effect once party and ideology are in
# the persona.
# LIMIT: the headlines are RSS titles only; nobody has read the articles (the collection
# script says so). Treat the news run as an experiment, not as the main result.


#### Workspace setup ####
import argparse

import numpy as np
import pandas as pd

import config

ap = argparse.ArgumentParser()
ap.add_argument("--designs", default="B", help="comma list from B,C")
ap.add_argument("--news", action=argparse.BooleanOptionalAction, default=config.NEWS_RUN,
                help="include the news (default: on for dated runs, off for RUN=nonews)")
args = ap.parse_args()
designs = [d.strip().upper() for d in args.designs.split(",")]
assert set(designs) <= {"B", "C"}, "designs must be B and/or C"

INSTRUCTIONS = (
    "You are taking part in a survey about the November 3, 2026 U.S. elections. "
    "Answer as the person described below would, in the first person. "
    "Do not say that you are an AI model."
)


def news_block(state_name):
    """National and state headlines as a short list. Empty text if no news file."""
    if not config.NEWS_FILE.exists():
        return ""
    news = pd.read_csv(config.NEWS_FILE)
    national = news[news["level"] == "national"].sort_values("published_at_utc", ascending=False).head(5)
    local = news[(news["level"] == "state") & (news["state"] == state_name)].sort_values(
        "published_at_utc", ascending=False).head(3)
    lines = [f"- {r['title']} ({r['source']})" for _, r in pd.concat([national, local]).iterrows()]
    if not lines:
        return ""
    return f"Recent news headlines (collected {config.AS_OF}; titles only):\n" + "\n".join(lines)


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
news_by_state = {s: news_block(config.STATE_NAME[s]) for s in nominees.index} if args.news else {}
# Stop instead of silently building news prompts that contain no news (paid calls would be wasted)
if args.news and not any(news_by_state.values()):
    raise SystemExit(f"Run {config.RUN} needs news, but no news text was found "
                     f"(expected {config.NEWS_SUMMARY_FILE}). Add the summary, or use RUN=nonews.")
print(f"Run: {config.RUN}  (news={args.news}; files in {config.RUN_DIR})")

rows = []
for _, p in personas.iterrows():
    race = nominees.loc[p["state"]].copy()
    left_first = bool(rng.integers(0, 2))    # one random order per persona, kept for all designs
    for design in designs:
        question = question_b(race, left_first) if design == "B" else question_c(race, left_first)
        parts = [INSTRUCTIONS]
        if args.news and news_by_state.get(p["state"]):
            parts.append(news_by_state[p["state"]])
        parts += [p["persona_text"], question]
        rows.append({
            "prompt_id": f"{p['persona_id']}|{design}|{config.RUN}",
            "run": config.RUN, "persona_id": p["persona_id"], "state": p["state"], "design": design,
            "news": bool(args.news), "left_first": left_first,
            "prompt_text": "\n\n".join(parts),
        })

prompts = pd.DataFrame(rows)

#### Checks and save ####
assert prompts["prompt_id"].is_unique
assert prompts["news"].eq(args.news).all()
assert prompts["prompt_text"].str.len().gt(100).all()
# News headlines may name politicians, so only the persona text is checked for the 2024 vote
assert not personas["persona_text"].str.contains("Trump|Harris", case=False).any(), "2024 vote leaked into a persona"

config.write_table(prompts, config.PROMPTS_FILE)
print(f"Saved {len(prompts)} prompts ({', '.join(designs)}; news={args.news}) to {config.PROMPTS_FILE}")
print("\nExample prompt:\n" + "-" * 60 + "\n" + prompts["prompt_text"].iloc[0] + "\n" + "-" * 60)
