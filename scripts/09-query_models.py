#### Preamble ####
# Purpose: Sends every prompt to each model on OpenRouter and saves the raw text answers,
#          with the exact model id returned, the time, and the token counts.
# Author: (group)
# Date: 22 October 2026
# License: MIT
# Pre-requisites: Run scripts/08-build_prompts.py first. Put your key in a file called .env in
#   the project root as one line:  OPENROUTER_API_KEY=sk-or-...   and add .env to .gitignore.
#   Needs: pip install requests pandas pyarrow
# Any other information needed? Run from the project root.
#   python scripts/09-query_models.py --limit 4      <- first try: 4 real calls
#   python scripts/09-query_models.py                <- everything not yet answered
#   python scripts/09-query_models.py --mock         <- no internet, no key: fake answers to
#                                                       test scripts 10-13
#
# FREE-MODEL LIMITS (OpenRouter docs, checked 22 Oct 2026): 20 requests per minute, and
# 50 per day unless you have bought at least 10 credits (then 1,000 per day). The test run
# is 175 personas x 2 models = 350 calls, so under the 50 limit it takes about 7 days. The
# script keeps what it has saved, skips finished prompts, and stops cleanly when the daily
# limit is hit, so you can run it again the next day. --sleep 3.2 keeps it under 20 per minute.
#
# WHY several models: different makers have different biases, so the paper can say whether the
# result depends on the model (Murray & Alexander 2026 used four).
# WHY we save model_returned and queried_at_utc: the disclosure checklist asks for exact model
# ids and query dates, and models change under the same name (Bisbee et al. 2024).
# NOTE: "stealth" models on OpenRouter are anonymous and can log prompts. Our prompts hold only
# invented survey personas, so this is acceptable, but say so in the paper. They can also be
# renamed or withdrawn without notice.
# LIMIT: reasoning models can spend their token budget thinking and return empty text. Those
# rows are saved with error = "empty_content", and the next run tries them again.


#### Workspace setup ####
import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import config

URL = "https://openrouter.ai/api/v1/chat/completions"

ap = argparse.ArgumentParser()
ap.add_argument("--models", default=",".join(config.MODELS), help="short names from config.MODELS")
ap.add_argument("--repeats", type=int, default=config.N_REPEATS)
ap.add_argument("--limit", type=int, default=None, help="stop after this many new calls")
ap.add_argument("--sleep", type=float, default=3.2, help="seconds between calls")
ap.add_argument("--max-tokens", type=int, default=800)
ap.add_argument("--mock", action="store_true", help="fake answers, no internet")
args = ap.parse_args()


class StopRun(Exception):
    """Raised for problems where more calls cannot help (bad key, no credits, daily limit)."""


def call_openrouter(key, model_id, prompt_text):
    """One chat call. Returns a dict; never raises for ordinary failures."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "X-Title": "Senate silicon sampling class project"}
    body = {"model": model_id, "max_tokens": args.max_tokens,
            "messages": [{"role": "user", "content": prompt_text}]}
    out = {"response_text": "", "model_returned": "", "input_tokens": None,
           "output_tokens": None, "error": ""}
    for attempt in range(4):
        try:
            r = requests.post(URL, headers=headers, json=body, timeout=120)
        except requests.RequestException as exc:
            out["error"] = f"network: {exc}"
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 401:
            raise StopRun("OpenRouter rejected the key (401). Check .env.")
        if r.status_code == 402:
            raise StopRun("OpenRouter says credits are needed (402).")
        if r.status_code == 429:
            text = r.text[:300]
            if "per-day" in text or "per day" in text or "daily" in text.lower():
                raise StopRun("Daily free-model limit reached. Run again tomorrow. " + text)
            out["error"] = "rate limited (429)"
            time.sleep(30 * (attempt + 1))
            continue
        if r.status_code >= 500:
            out["error"] = f"server error {r.status_code}"
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code != 200:
            out["error"] = f"http {r.status_code}: {r.text[:200]}"
            return out
        data = r.json()
        if "error" in data:
            out["error"] = f"api error: {str(data['error'])[:200]}"
            return out
        choice = data["choices"][0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        out.update(response_text=text.strip(), model_returned=data.get("model", ""),
                   input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                   error="" if text.strip() else "empty_content")
        return out
    return out


def mock_answer(prompt_text, design, model, seed_text):
    """Fake answer shaped like a real one. Used only to test the rest of the pipeline."""
    rng = np.random.default_rng(int(hashlib.md5((seed_text + model).encode()).hexdigest()[:8], 16))
    persona = prompt_text.split("\n\n")[-2]
    if re.search(r"strong Democrat|Democrat, but|leans toward the Democratic", persona):
        side = "left"
    elif re.search(r"strong Republican|Republican, but|leans toward the Republican", persona):
        side = "right"
    else:
        side = "middle"
    question = prompt_text.split("\n\n")[-1]
    pairs = re.findall(r"(?:are|and) ([^()]+?) \((Democratic|Independent|Republican)\)", question)
    left_name, left_desc, rep_name = "the left candidate", "left-leaning", "the Republican"
    for name, party in pairs:
        if party == "Republican":
            rep_name = name.strip()
        else:
            left_name, left_desc = name.strip(), party.lower()
    u = rng.uniform()
    if design == "C":
        base = {"left": [70, 20, 5, 5], "right": [20, 70, 5, 5], "middle": [35, 35, 10, 20]}[side]
        noisy = np.maximum(np.array(base) + rng.integers(-8, 9, 4), 0)
        noisy = np.round(noisy * 100 / noisy.sum()).astype(int)
        who_is_a = re.search(r"A = (.+?), how many for B", question)
        a_is_right = bool(who_is_a) and who_is_a.group(1).strip() == rep_name
        a, b = (noisy[1], noisy[0]) if a_is_right else (noisy[0], noisy[1])
        return json.dumps({"A": int(a), "B": int(b), "other": int(noisy[2]), "not_voting": int(noisy[3])})
    if u < 0.03:
        return ""
    if u < 0.07:
        return "I'm sorry, but I can't take a side in an election."
    if u < 0.12:
        return "I haven't decided yet, and I might not vote at all."
    if side == "left":
        return (f"I plan to vote for {left_name}, the {left_desc} candidate. "
                "I agree with that side on most issues." if u < 0.7 else
                f"I am leaning toward {left_name}, the {left_desc} candidate, though I am not fully sure.")
    if side == "right":
        return (f"I will vote for {rep_name}, the Republican candidate. That is who I have supported before."
                if u < 0.7 else f"I lean toward {rep_name}, the Republican candidate, but I am still thinking.")
    return (f"I am undecided, but I may lean toward {left_name}." if u < 0.5
            else f"I am torn, but I may lean toward {rep_name}, the Republican candidate.")


#### Work out what still needs asking ####
prompts = config.read_table(config.PROMPTS_FILE)
short_names = [m.strip() for m in args.models.split(",")]
for m in short_names:
    assert m in config.MODELS, f"Unknown model {m}; known: {list(config.MODELS)}"

try:
    existing = config.read_table(config.RESPONSES_FILE)
    existing["error"] = existing["error"].fillna("")
    kept = existing[existing["error"] == ""]            # failed rows are tried again
except (FileNotFoundError, OSError):
    kept = pd.DataFrame()

done = set(zip(kept["prompt_id"], kept["model"], kept["repeat"])) if len(kept) else set()
todo = [(rep, m, p) for rep in range(1, args.repeats + 1) for m in short_names
        for p in prompts.itertuples() if (p.prompt_id, m, rep) not in done]
print(f"{len(prompts)} prompts x {len(short_names)} models x {args.repeats} repeats; "
      f"{len(done)} already answered, {len(todo)} to do")
if args.limit is not None:
    todo = todo[: args.limit]
    print(f"--limit: doing {len(todo)} this run")

key = None
if not args.mock:
    key = config.get_api_key()
    if not key:
        sys.exit("No API key found. Put OPENROUTER_API_KEY=... in a .env file in the project root.")

#### Ask ####
new_rows, errors_in_a_row = [], 0


def save():
    frame = pd.concat([kept, pd.DataFrame(new_rows)], ignore_index=True) if len(kept) else pd.DataFrame(new_rows)
    if len(frame):
        config.write_table(frame, config.RESPONSES_FILE)


try:
    for i, (rep, short, p) in enumerate(todo, start=1):
        if args.mock:
            result = {"response_text": mock_answer(p.prompt_text, p.design, short, p.prompt_id),
                      "model_returned": "MOCK-" + config.MODELS[short], "input_tokens": 0,
                      "output_tokens": 0, "error": ""}
            if not result["response_text"]:
                result["error"] = "empty_content"
        else:
            result = call_openrouter(key, config.MODELS[short], p.prompt_text)
            time.sleep(args.sleep)
        new_rows.append({
            "prompt_id": p.prompt_id, "persona_id": p.persona_id, "state": p.state, "design": p.design,
            "model": short, "model_requested": config.MODELS[short], "repeat": rep,
            "queried_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **result})
        errors_in_a_row = errors_in_a_row + 1 if result["error"] and result["error"] != "empty_content" else 0
        if errors_in_a_row >= 5:
            raise StopRun("Five failed calls in a row: " + result["error"])
        if i % 10 == 0:
            save()
            print(f"  {i}/{len(todo)} done", flush=True)
except StopRun as stop:
    print("\nSTOPPED:", stop)
finally:
    save()

if new_rows:
    got = pd.DataFrame(new_rows)
    print(f"\nThis run: {len(got)} calls, {int((got['error'] != '').sum())} with an error")
    print(got.groupby("model")["model_returned"].agg(lambda s: sorted(set(s))).to_string())
print(f"Saved to {config.RESPONSES_FILE}")
