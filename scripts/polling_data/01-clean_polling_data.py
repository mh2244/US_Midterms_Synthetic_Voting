#### Preamble ####
# Purpose: Selects useful polling columns, standardizes dates, and retains
#          2026 general-election Senate polls from the downloaded dataset.
# Author: Jingxuan Feng
# Date: 24 October 2026
# Contact: amelia.f.feng@outlook.com
# License: MIT
# Pre-requisites: Python 3.10+; run 00-download_polling_data.py first.
#                 Uses only the Python standard library.

"""Select useful columns and retain 2026-cycle general-election Senate polls.

Run: python3 clean_pollresults.py
Or:  python3 clean_pollresults.py path/to/senate_polls_all.csv
Uses only the Python standard library. The original download is not modified.
One output row remains one answer within a poll question, not one entire poll.
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = ROOT / "01-raw_data" / "polls" / "pollresults"
OUTPUT_FILE = ROOT / "02-analysis_data" / "polls" / "senate_polls_2026_general.csv"

# Keep a compact view with IDs to trace every answer back to the raw data.
COLUMNS = {
    "poll_pollId": "poll_id",
    "question_questionId": "question_id",
    "poll_state": "state",
    "poll_pollster": "pollster",
    "poll_startDate": "start_date",
    "poll_endDate": "end_date",
    "question_sampleSize": "sample_size",
    "question_population": "population",
    "question_subpopulation": "subpopulation",
    "result_party": "party",
    "result_pct": "pct",
    "poll_hypothetical": "hypothetical",
    "result_rankedChoiceRound": "ranked_choice_round",
}


def iso_date(value):
    """Convert fieldwork dates to YYYY-MM-DD without assuming the election year."""
    if not value:
        return ""
    for pattern in ("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, pattern).date().isoformat()
        except ValueError:
            pass
    raise ValueError(f"Unrecognized date: {value!r}")


def clean(input_file, output_file):
    with input_file.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = set(COLUMNS) | {"poll_cycle", "question_stage", "result_candidateName", "result_answer", "poll_url"}
        missing -= set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing expected input columns: {', '.join(sorted(missing))}")
        rows = []
        for original in reader:
            if original["poll_cycle"].strip() != "2026":
                continue
            if original["question_stage"].strip().lower() != "general":
                continue
            row = {new: original[old] for old, new in COLUMNS.items()}
            # Preserve non-candidate answers such as Don't know and Would not vote.
            row["response_name"] = original["result_candidateName"] or original["result_answer"]
            row["source_url"] = next((original.get(key) for key in (
                "poll_urlTopline", "poll_url", "poll_urlArticle", "poll_urlCrosstab"
            ) if original.get(key)), "")
            for key in ("start_date", "end_date"):
                row[key] = iso_date(row[key].strip())
            rows.append(row)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    columns = list(COLUMNS.values()) + ["response_name", "source_url"]
    with output_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} answer rows from {len({r['poll_id'] for r in rows})} polls")
    print(f"States covered: {len({r['state'] for r in rows if r['state']})}")
    print(f"Output: {output_file}")
    print("No recency, candidate-matchup, subgroup or ranked-choice-round filtering was applied.")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="Downloaded senate_polls_all.csv")
    args = parser.parse_args()
    input_file = args.input
    if input_file is None:
        downloads = sorted(RAW_DIR.glob("*/senate_polls_all.csv"))
        if not downloads:
            raise FileNotFoundError("Run 00-download_polling_data.py first or pass the input CSV path.")
        input_file = downloads[-1]
    output_file = OUTPUT_FILE
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if input_file.resolve() == output_file.resolve():
        raise ValueError("Input and output paths must be different.")
    print(f"Input: {input_file}")
    clean(input_file, output_file)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(f"Cleaning stopped: {error}") from None
