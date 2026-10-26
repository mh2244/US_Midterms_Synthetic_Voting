#### Preamble ####
# Purpose: Turns each model answer into three numbers: the probability the persona votes for
#          the left candidate (p_left), the Republican (p_rep), or for someone else / does not
#          vote / is undecided (p_other).
#          Design B answers (free text) are scored with Semantic Similarity Rating (SSR).
#          Design C answers (percentages) are only parsed.
# Author: (group)
# Date: 22 October 2026
# License: MIT
# Pre-requisites: Run scripts/09-query_models.py first (or with --mock).
#   For the local embedder: pip install sentence-transformers   (downloads a small model once)
#   For Gemini Embedding 2 (the paper's model): --embedder openrouter  (about $0.20 per million tokens)
# Any other information needed? Run from the project root.
#   python scripts/10-score_responses.py                      <- local embedding model
#   python scripts/10-score_responses.py --embedder mock      <- offline plumbing test only
#
# HOW SSR WORKS here (Maier et al. 2025; Heath & Alexander 2026):
#   1. Turn the answer and five anchor statements into embeddings (lists of numbers that
#      capture meaning).
#   2. Compare the answer with each anchor (cosine similarity).
#   3. Rescale the five similarities to 0-1, divide by a temperature T, and apply softmax.
#      This gives five probabilities. Anchors 1+2 = left, 4+5 = Republican, 3 = other.
# WHY SSR and not "ask for a number": asking for a number gives answers that vary too little
# (Heath & Alexander 2026). SSR keeps the variation. It does not remove the bias: models still
# lean too far one way, which is why the forecast script compares with polls.
# LIMITS we should write in the paper:
#   - SSR was tested on 0-100 ratings; applying it to vote choice is our adaptation.
#   - Heath & Alexander used asymmetric embeddings (Gemini Embedding 2). Our default local
#     model is symmetric and smaller, so scores will differ.
#   - Embeddings capture topic closeness more than stance ("I would never vote for X" can sit
#     close to "I will vote for X"). READ ~30 ANSWERS BY HAND next to their scores.
#   - The anchors name the real candidates of each race; changing the anchor wording changes
#     the shares, so try a second wording and report the difference.


#### Workspace setup ####
import argparse
import json
import re
import time
import zlib

import numpy as np
import pandas as pd
import requests

import config

ap = argparse.ArgumentParser()
ap.add_argument("--embedder", default=config.EMBEDDER, choices=["local", "openrouter", "mock"])
ap.add_argument("--temperature", type=float, default=config.SSR_TEMPERATURE)
args = ap.parse_args()

REFUSAL = re.compile(
    r"\b(as an ai|i('m| am) (sorry|unable)|i (cannot|can't) (take a side|answer (that|this)|provide an opinion|"
    r"share (my|an) opinion|participate))\b", re.I)


#### Embedders ####
def embed_local(texts, as_query=False):
    from sentence_transformers import SentenceTransformer     # imported here so it is optional
    model = SentenceTransformer(config.LOCAL_EMBED_MODEL)
    return np.asarray(model.encode(texts, normalize_embeddings=True))


def embed_openrouter(texts, as_query=False, chunk=100):
    """Send the texts in chunks: Gemini accepts at most 100 texts per request (HTTP 400 above that).
    as_query=True marks the model's answers; False marks the anchors (see config for the prefixes).
    Each chunk is retried up to 4 times, so one failed request does not stop a full run."""
    key = config.get_api_key()
    assert key, "No OPENROUTER_API_KEY found (see script 09)"
    prefix = config.EMBED_QUERY_PREFIX if as_query else config.EMBED_DOC_PREFIX
    texts = [prefix + t for t in texts]
    pieces = []
    for start in range(0, len(texts), chunk):
        body = {"model": config.OPENROUTER_EMBED_MODEL, "input": texts[start:start + chunk]}
        if config.EMBED_DIMENSIONS:
            body["dimensions"] = config.EMBED_DIMENSIONS
        for attempt in range(4):
            try:
                r = requests.post("https://openrouter.ai/api/v1/embeddings",
                                  headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
                if r.status_code == 200:
                    break
                problem = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code in (400, 401, 402):
                    raise SystemExit(f"Embedding request refused ({problem})")   # retrying cannot help
            except requests.RequestException as exc:
                problem = f"network: {exc}"
            time.sleep(10 * (attempt + 1))
        else:
            raise SystemExit(f"Embedding failed 4 times ({problem})")
        data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
        pieces.append(np.asarray([d["embedding"] for d in data], dtype=np.float32))
    vecs = np.vstack(pieces)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def embed_mock(texts, as_query=False, dim=512):
    """Word-count vectors from hashed 4-letter word starts. Only for testing the plumbing."""
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z]+", t.lower()):
            out[i, zlib.crc32(w[:4].encode()) % dim] += 1
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norm == 0, 1, norm)


EMBED = {"local": embed_local, "openrouter": embed_openrouter, "mock": embed_mock}[args.embedder]


def ssr_probabilities(sims, temperature):
    """Five similarities -> five probabilities. Example:
        ssr_probabilities([0.2, 0.5, 0.3, 0.1, 0.0], 0.25)"""
    s = np.asarray(sims, dtype=float)
    spread = s.max() - s.min()
    s = (s - s.min()) / spread if spread > 0 else np.zeros_like(s)   # min-max to 0-1
    z = np.exp(s / temperature)
    return z / z.sum()


def anchors_for(race):
    left_desc = "Democratic" if race["left_party"] == "Democratic" else race["left_party"].lower()
    left = f"the {left_desc} candidate, {race['left_name']}"
    right = f"the Republican candidate, {race['rep_name']}"
    return [
        f"I will definitely vote for {left}.",
        f"I lean toward voting for {left}.",
        "I am undecided, I will vote for someone else, or I will not vote.",
        f"I lean toward voting for {right}.",
        f"I will definitely vote for {right}.",
    ]


def parse_shares(text, left_first):
    """Design C: read the JSON, accept totals of 90-110, rescale to 100 (same rule as the
    Murray & Alexander paper). Returns (p_left, p_rep, p_other) or None."""
    match = re.search(r"\{.*?\}", text, re.S)
    if not match:
        return None
    try:
        d = json.loads(match.group(0))
        a, b, other, none = (float(d[k]) for k in ("A", "B", "other", "not_voting"))
    except (ValueError, KeyError, TypeError):
        return None
    total = a + b + other + none
    if not 90 <= total <= 110:
        return None
    a, b, other, none = (v * 100 / total for v in (a, b, other, none))
    left, right = (a, b) if left_first else (b, a)
    return left / 100, right / 100, (other + none) / 100


#### Score ####
responses = config.read_table(config.RESPONSES_FILE)
prompts = config.read_table(config.PROMPTS_FILE)[["prompt_id", "left_first"]]
nominees = pd.read_csv(config.NOMINEES_FILE).set_index("state")
responses = responses.merge(prompts, on="prompt_id", how="left")
responses["response_text"] = responses["response_text"].fillna("")
responses["error"] = responses["error"].fillna("")

# Status first: empty and refused answers are counted, not scored
responses["status"] = "ok"
is_refusal = responses["response_text"].map(lambda t: bool(REFUSAL.search(t)))
responses.loc[is_refusal, "status"] = "refusal"
responses.loc[responses["response_text"].str.strip() == "", "status"] = "empty"
responses.loc[(responses["error"] != "") & (responses["error"] != "empty_content"), "status"] = "error"

responses[["p_left", "p_rep", "p_other"]] = np.nan

# Design B: SSR
is_b = (responses["design"] == "B") & (responses["status"] == "ok")
if is_b.any():
    texts = responses.loc[is_b, "response_text"].tolist()
    anchor_rows = {s: anchors_for(nominees.loc[s]) for s in responses.loc[is_b, "state"].unique()}
    all_anchor_text = sorted({a for lst in anchor_rows.values() for a in lst})
    answer_texts = sorted(set(texts))
    print(f"Embedding {len(answer_texts)} unique answers and {len(all_anchor_text)} anchors "
          f"with the '{args.embedder}' embedder ...")
    answer_vecs = dict(zip(answer_texts, EMBED(answer_texts, as_query=True)))        # answers = queries
    anchor_vecs = dict(zip(all_anchor_text, EMBED(all_anchor_text, as_query=False)))  # anchors = documents

    for idx, row in responses[is_b].iterrows():
        anchors = anchor_rows[row["state"]]
        sims = [float(answer_vecs[row["response_text"]] @ anchor_vecs[a]) for a in anchors]
        p = ssr_probabilities(sims, args.temperature)
        responses.loc[idx, ["p_left", "p_rep", "p_other"]] = [p[0] + p[1], p[3] + p[4], p[2]]

# Design C: parse
for idx, row in responses[(responses["design"] == "C") & (responses["status"] == "ok")].iterrows():
    parsed = parse_shares(row["response_text"], bool(row["left_first"]))
    if parsed is None:
        responses.loc[idx, "status"] = "bad_shares"
    else:
        responses.loc[idx, ["p_left", "p_rep", "p_other"]] = parsed

# A readable label, like the vote_choice column of the simulated responses
labels = responses[["p_left", "p_rep", "p_other"]].to_numpy()
best = np.array([np.nan if np.isnan(r).any() else int(np.argmax(r)) for r in labels])
left_party = responses["state"].map(nominees["left_party"])
responses["vote_choice"] = pd.Series(best).map({0: "left", 1: "Republican", 2: "Undecided"}).where(~np.isnan(best))
responses.loc[responses["vote_choice"] == "left", "vote_choice"] = left_party
responses["ssr_temperature"] = args.temperature
responses["embedder"] = {"local": config.LOCAL_EMBED_MODEL, "openrouter": config.OPENROUTER_EMBED_MODEL,
                         "mock": "mock"}[args.embedder]

#### Checks and save ####
scored = responses[responses["p_left"].notna()]
sums = scored[["p_left", "p_rep", "p_other"]].sum(axis=1)
assert np.isclose(sums, 1).all(), "Probabilities do not add up to 1"

config.write_table(responses.drop(columns=["response_text"]), config.SCORES_FILE)
responses[["prompt_id", "model", "repeat", "status", "response_text", "p_left", "p_rep", "p_other"]].to_csv(
    config.RUN_DIR / "scores_with_text.csv", index=False)   # open this and READ the answers

print(f"\nSaved {len(responses)} rows to {config.SCORES_FILE}")
print("\nStatus by model and design (anything but 'ok' is not scored):")
print(pd.crosstab([responses["design"], responses["model"]], responses["status"]).to_string())
print("\nMean probabilities by model and design:")
print(scored.groupby(["design", "model"])[["p_left", "p_rep", "p_other"]].mean().round(3).to_string())
print(f"\nOpen {config.RUN_DIR / 'scores_with_text.csv'} and read about 30 answers next to their scores.")