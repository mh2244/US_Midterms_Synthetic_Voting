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
#   python scripts/09-query_models.py --batch-size 10 --sleep 0   <- paid models: 10 requests at once
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
# NOTE: free models can log prompts and can be withdrawn without notice (the stealth model we
# first chose was). Our prompts hold only invented survey personas, so logging is acceptable,
# but say so in the paper.
# LIMIT: reasoning models can spend their token budget thinking and return empty text. Those
# rows are saved with error = "empty_content", and the next run tries them again.
# SAFETY (added 2026-10-26): a malformed response (bad JSON, no "choices") or any other
# unexpected error becomes an error row for that call only, so it can never stop the run or
# lose the other paid answers of the same batch. An answer cut off by max_tokens
# (finish_reason "length") is kept for inspection but marked as an error, so it is not scored.
# Every failed call is also appended to failed_calls.csv in the run folder; the responses file
# keeps only the latest attempt, so that log is what the paper's failure counts come from.


#### Workspace setup ####
import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import config

URL = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")   # override is for testing only

ap = argparse.ArgumentParser()
ap.add_argument("--models", default=",".join(config.MODELS), help="short names from config.MODELS")
ap.add_argument("--repeats", type=int, default=config.N_REPEATS)
ap.add_argument("--limit", type=int, default=None, help="stop after this many new calls")
ap.add_argument("--sleep", type=float, default=3.2,
                help="minimum average seconds per call (3.2 = under 20 a minute); use 0 on a paid model")
ap.add_argument("--batch-size", type=int, default=1,
                help="send this many requests at the same time (1 = one at a time)")
ap.add_argument("--max-tokens", type=int, default=800)
ap.add_argument("--mock", action="store_true", help="fake answers, no internet")
args = ap.parse_args()
if args.repeats < 1 or (args.limit is not None and args.limit < 1):
    ap.error("--repeats and --limit must be positive")


class StopRun(Exception):
    """Raised for problems where more calls cannot help (bad key, no credits, daily limit)."""


def call_openrouter(key, model_id, prompt_text):
    """One chat call. Returns a dict; never raises for ordinary failures."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "X-Title": "Senate silicon sampling class project"}
    body = {"model": model_id, "max_tokens": args.max_tokens,
            "messages": [{"role": "user", "content": prompt_text}]}
    settings = getattr(config, "REASONING", None) or {}
    short = {v: k for k, v in config.MODELS.items()}.get(model_id)
    reasoning = settings.get(short, settings.get("default"))
    if reasoning is not None:
        body["reasoning"] = reasoning   # see config.py: thinking off (or minimal) to save output tokens
    out = {"response_text": "", "model_returned": "", "input_tokens": None,
           "output_tokens": None, "finish_reason": "", "error": ""}
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
            out["error"] = "rate limited (429): " + " ".join(text.split())[:220]
            time.sleep(30 * (attempt + 1))
            continue
        if r.status_code >= 500:
            out["error"] = f"server error {r.status_code}"
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code != 200:
            out["error"] = f"http {r.status_code}: {r.text[:200]}"
            return out
        try:
            data = r.json()
            if not isinstance(data, dict):
                raise TypeError(f"response is a {type(data).__name__}, not an object")
            if "error" in data:
                out["error"] = f"api error: {str(data['error'])[:200]}"
                return out
            choice = data["choices"][0]
            text = (choice.get("message") or {}).get("content") or ""
            finish = str(choice.get("finish_reason") or "")
            usage = data.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            # Bad JSON or an unexpected shape: try this call again, then give up on it alone
            out["error"] = f"bad response ({type(exc).__name__}): " + " ".join(r.text[:150].split())
            time.sleep(5 * (attempt + 1))
            continue
        text = text.strip() if isinstance(text, str) else ""
        if not text:
            error = "empty_content"
        elif finish == "length":
            error = "truncated (finish_reason=length)"     # cut off by max_tokens; not scored
        elif finish == "error":
            error = "provider error (finish_reason=error)"  # stream broke mid-answer; not scored
        else:
            error = ""
        out.update(response_text=text, model_returned=data.get("model", ""),
                   input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                   finish_reason=finish, error=error)
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
print(f"Run: {config.RUN}  (files in {config.RUN_DIR})")
prompts = config.read_table(config.PROMPTS_FILE)
short_names = [m.strip() for m in args.models.split(",")]
for m in short_names:
    assert m in config.MODELS, f"Unknown model {m}; known: {list(config.MODELS)}"

try:
    existing = config.read_table(config.RESPONSES_FILE)
    existing["error"] = existing["error"].fillna("")
    valid = existing["error"].eq("") & existing["prompt_id"].isin(prompts["prompt_id"])
    valid &= existing["response_text"].fillna("").str.strip().ne("")
    # Rows for models or repeats not asked in THIS command are kept (not re-asked), so running
    # one model with --models, or fewer --repeats, never deletes answers already paid for.
    valid &= existing["repeat"].ge(1)
    valid &= existing["model_requested"].eq(existing["model"].map(config.MODELS))
    valid &= existing["model_returned"].fillna("").str.startswith("MOCK-").eq(args.mock)
    kept = existing.loc[valid].copy()  # exclude obsolete experiments and mock/real mixtures
except (FileNotFoundError, OSError):
    kept = pd.DataFrame()

done = set(zip(kept["prompt_id"], kept["model"], kept["repeat"])) if len(kept) else set()
todo = [(rep, m, p) for rep in range(1, args.repeats + 1) for m in short_names
        for p in prompts.itertuples() if (p.prompt_id, m, rep) not in done]
print(f"{len(prompts)} prompts x {len(short_names)} models x {args.repeats} repeats; "
      f"{len(done)} already answered, {len(todo)} to do")
required_calls = len(todo)  # count all unfinished work before applying a pilot limit
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
FAILED_LOG = config.RUN_DIR / "failed_calls.csv"
logged = 0                                  # how many of new_rows are already in the failure log
stopped = False


def save():
    """Write all answers so far, and append new failed calls to the failure log."""
    global logged
    frame = pd.concat([kept, pd.DataFrame(new_rows)], ignore_index=True) if len(kept) else pd.DataFrame(new_rows)
    if len(frame):
        config.write_table(frame, config.RESPONSES_FILE)
    fresh = pd.DataFrame(new_rows[logged:])
    if len(fresh):
        failed = fresh[fresh["error"] != ""].drop(columns=["response_text"], errors="ignore")
        if len(failed):
            FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
            failed.to_csv(FAILED_LOG, mode="a", header=not FAILED_LOG.exists(), index=False)
    logged = len(new_rows)


def ask_one(item):
    """Get one result (mock or real) for one to-do item."""
    rep, short, p = item
    if args.mock:
        result = {"response_text": mock_answer(p.prompt_text, p.design, short, p.prompt_id),
                  "model_returned": "MOCK-" + config.MODELS[short], "input_tokens": 0,
                  "output_tokens": 0, "finish_reason": "stop", "error": ""}
        if not result["response_text"]:
            result["error"] = "empty_content"
        return result
    return call_openrouter(key, config.MODELS[short], p.prompt_text)


try:
    done_count, since_save = 0, 0
    batch = max(1, args.batch_size)
    with ThreadPoolExecutor(max_workers=batch) as pool:
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            t0 = time.time()
            futures = [pool.submit(ask_one, item) for item in chunk]    # the whole group is sent at once
            stop_signal = None
            for (rep, short, p), fut in zip(chunk, futures):            # results are matched back by position
                try:
                    result = fut.result()
                except StopRun as stop:
                    stop_signal = stop                                   # keep the answers that did arrive
                    continue
                except Exception as exc:                                 # never lose the rest of the batch
                    result = {"response_text": "", "model_returned": "", "input_tokens": None,
                              "output_tokens": None, "finish_reason": "",
                              "error": f"unexpected ({type(exc).__name__}): {str(exc)[:150]}"}
                new_rows.append({
                    "prompt_id": p.prompt_id, "run": config.RUN, "persona_id": p.persona_id,
                    "state": p.state, "design": p.design,
                    "model": short, "model_requested": config.MODELS[short], "repeat": rep,
                    "queried_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **result})
                real_error = result["error"] and result["error"] != "empty_content"
                errors_in_a_row = errors_in_a_row + 1 if real_error else 0
                if errors_in_a_row >= 5 and stop_signal is None:
                    stop_signal = StopRun("Five failed calls in a row: " + result["error"])
            done_count += len(chunk)
            since_save += len(chunk)
            if since_save >= 10:
                save()
                print(f"  {done_count}/{len(todo)} done", flush=True)
                since_save = 0
            if stop_signal is not None:
                raise stop_signal
            if not args.mock:                                            # stay under the per-minute limit
                time.sleep(max(0.0, len(chunk) * args.sleep - (time.time() - t0)))
except StopRun as stop:
    stopped = True
    print("\nSTOPPED:", stop)
finally:
    save()

if new_rows:
    got = pd.DataFrame(new_rows)
    print(f"\nThis run: {len(got)} calls, {int((got['error'] != '').sum())} with an error")
    print(got.groupby("model")["model_returned"].agg(lambda s: sorted(set(s))).to_string())
    problems = got[got["error"] != ""]
    if len(problems):
        print("\nERROR MESSAGES (first of each kind, by model):")
        for (model, err), n in problems.groupby(["model", "error"]).size().items():
            print(f"  {model}: {err}   (x{n})")
print(f"Saved to {config.RESPONSES_FILE}")
# A partial run is saved for resuming, but must not trigger downstream forecasting.
completed_calls = sum(not row["error"] and bool(row["response_text"].strip()) for row in new_rows)
remaining_calls = required_calls - completed_calls
if stopped or remaining_calls:
    print(f"INCOMPLETE: {remaining_calls} calls still need successful answers. "
          "Progress saved; rerun 09 with the same prompts to continue.")
    sys.exit(1)
print("COMPLETE: all requested prompts/models/repeats have successful answers.")
