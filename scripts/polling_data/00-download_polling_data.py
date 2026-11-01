#### Preamble ####
# Purpose: Downloads the complete Senate polling dataset from the PollResults API
#          and saves the raw JSON and flattened CSV for our 2026 Senate project.
# Author: Jingxuan Feng
# Date: 24 October 2026
# Contact: amelia.f.feng@outlook.com
# License: MIT
# Pre-requisites: Python 3.10+ and system curl; no additional Python packages.

"""Download the complete Senate dataset from the documented PollResults API.

Run: python3 download_pollresults.py
Requires Python 3.10+ and system curl; no Python packages or API key needed.
API documentation: https://pollresults.org/api/
Data provider: PollResults.org, using New York Times polling data (CC BY 4.0).
Downloading data does not independently verify the underlying poll reports.
"""
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

API = "https://api.pollresults.org/v1.0/politics/results/senate"
ROOT = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = ROOT / "01-raw_data" / "polls" / "pollresults"      # one timestamped folder per download


def get_page(cursor=None):
    params = {"limit": 100}
    if cursor:
        params["cursor"] = cursor
    url = API + "?" + urlencode(params)
    response = subprocess.run([
        "curl", "-L", "--fail", "--silent", "--show-error",
        "--connect-timeout", "15", "--max-time", "60",
        "--retry", "2", "--retry-delay", "2", url,
    ], capture_output=True)
    if response.returncode:
        raise RuntimeError(response.stderr.decode("utf-8", errors="replace").strip())
    data = json.loads(response.stdout)
    if not isinstance(data.get("results"), list):
        raise ValueError("Unexpected API response: missing results array.")
    return data


def flatten(polls):
    """One CSV row per poll-question-answer; retain all returned field names."""
    rows = []
    for poll in polls:
        base = {"poll_" + k: v for k, v in poll.items() if k != "questions"}
        for question in poll.get("questions") or [{}]:
            fields = {"question_" + k: v for k, v in question.items() if k != "answers"}
            for answer in question.get("answers") or [{}]:
                results = {"result_" + k: v for k, v in answer.items()}
                rows.append({**base, **fields, **results})
    return rows


def save_csv(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                             else v for k, v in row.items()})


def main():
    started = datetime.now(timezone.utc)
    folder = RAW_DIR / started.strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True, exist_ok=True)
    polls, seen_ids, seen_cursors = [], set(), set()
    cursor, expected_total, metadata, page_number = None, None, {}, 0

    while True:
        page = get_page(cursor)
        page_number += 1
        if page_number == 1:
            expected_total = page.get("total")
            metadata = {k: v for k, v in page.items() if k not in ("results", "cursor", "count", "limit")}
        elif page.get("total") != expected_total:
            raise ValueError("API total changed during collection. Rerun to obtain a consistent download.")
        for poll in page["results"]:
            poll_id = poll.get("pollId")
            if not poll_id or poll_id in seen_ids:
                raise ValueError("Missing or repeated poll ID. Collection stopped instead of silently dropping data.")
            seen_ids.add(poll_id)
            polls.append(poll)
        print(f"Page {page_number}: collected {len(polls)} / {expected_total} polls.", flush=True)
        cursor = page.get("cursor")
        if not cursor:
            break
        if cursor in seen_cursors or not page["results"]:
            raise ValueError("Pagination did not advance; refusing to save an incomplete dataset.")
        seen_cursors.add(cursor)
        time.sleep(0.5)

    if expected_total is None or len(polls) != expected_total:
        raise ValueError(f"Incomplete download: expected {expected_total}, received {len(polls)}.")
    metadata.update(api_url=API, collected_at_utc=started.isoformat(),
                    completed_at_utc=datetime.now(timezone.utc).isoformat(), pages=page_number)
    (folder / "senate_polls_complete.json").write_text(
        json.dumps({"metadata": metadata, "results": polls}, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = flatten(polls)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    save_csv(folder / "senate_polls_all.csv", rows, columns)
    print(f"\nSaved {len(polls)} polls and {len(rows)} poll-question-answer rows to:\n{folder}")
    print("One poll can contain multiple questions and answers; CSV rows are not independent polls.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError) as error:
        raise SystemExit(f"Collection stopped: {error}") from None
