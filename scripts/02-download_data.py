"""
Preamble:

Purpose: Find and download 2024 Cooperative Election Study (CES) files from Harvard Dataverse.
Authors: Maggie, Coco, Amelia
Date: 19 October 2026 (paths updated 28 October 2026)
Contact: maggieh.huang@mail.utoronto.ca
License: MIT
Pre-requisites: Download the CES 2024 CSV directly by file id by running this in terminal:
    python scripts/02-download_data.py get --file-id 12050325
Output: data/01-raw_data/ces/ (the raw CES file is not committed; see the README)
"""

# Import libraries
import argparse  # Reads what you type after script name in the terminal (e.g. the file ID)
import re  # Parsing text
import sys  # Lets the script stop with an error message
from pathlib import Path  # Handling folder paths

import requests  # Communicates with websites

BASE = "https://dataverse.harvard.edu"
OUT_DIR = Path("data/01-raw_data/ces")  # Folder to save dataset to (raw data, not committed)
TIMEOUT = 60  # Give up after 60 seconds if the server does not answer
SESSION = requests.Session()  # Reusable connection that remembers settings

# Save these settings when communicating with Harvard dataverse website
SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (compatible; class-project-data-download/1.0)",
        "Accept": "*/*",
    }
)


# Takes query and integer
def search(query: str, n: int = 15):
    # SESSION.get() sends a request to the website
    r = SESSION.get(
        f"{BASE}/api/search", params={"q": query, "type": "dataset", "per_page": n}, timeout=TIMEOUT
    )

    # Stops with an error if server rejects request
    r.raise_for_status()

    # Converts reply into Python dictionary (i.e. lookup table). Go down 2 levels
    items = r.json()["data"]["items"]

    if not items:
        print("No datasets found. Try a different query.")
    for it in items:
        print(f"{it.get('name')}\n   {it.get('global_id')}  (published {it.get('published_at', '?')[:10]})")


# Ask website for list of files.
def get_files(doi: str):
    r = SESSION.get(f"{BASE}/api/datasets/:persistentId/", params={"persistentId": doi}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["data"]["latestVersion"]["files"]


def list_files(doi: str):
    for f in get_files(doi):
        df = f["dataFile"]
        # Right align ID in 10 characters and left align name in 55 characters
        print(f"{df['id']:>10}  {df['filename']:<55} {df.get('filesize', 0) / 1e6:8.1f} MB")


# Leading underscore means helper function, not meant to be called directly.
# Gets the file's real name from server's reply
def _name_from_headers(resp, file_id):
    cd = resp.headers.get("Content-Disposition", "")

    # Using re.search to get the name
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
    return m.group(1) if m else f"file_{file_id}"


def download(file_id: int, filename=None, original: bool = False):
    """Stream one file into data/. If filename is None, the server's name is used."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params = {"format": "original"} if original else {}
    # downloads dataset in pieces instead of all at once
    with SESSION.get(
        f"{BASE}/api/access/datafile/{file_id}", params=params, stream=True, timeout=TIMEOUT
    ) as r:
        r.raise_for_status()
        name = filename or _name_from_headers(r, file_id)
        dest = OUT_DIR / name

        # Checks whether the dataset is already in the output folder
        if dest.exists():
            print(f"Already have {dest}, skipping.")
            return dest
        # wb=write in binary mode. 1<<20 means 1 mb. Takes 1 MB at a time and writes to disk
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


# reading what user typed
def main():
    # Takes preamble and shows it when someone runs --help
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Has subcommands, store whichever one is chosen under cmd
    # subcommands get their own miniform
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    lst = sub.add_parser("list")
    lst.add_argument("doi")
    # defines get commands
    g = sub.add_parser("get")
    g.add_argument("doi", nargs="?", help="only needed with --match")
    g.add_argument("--match", help="download files whose name contains this text")
    g.add_argument("--file-id", type=int, help="download one file by id (no DOI needed)")
    g.add_argument(
        "--format",
        choices=["tab", "original"],
        default="tab",
        help="tab = as stored on Dataverse (default); original = as uploaded",
    )

    # reads typed command, check against form
    a = ap.parse_args()

    # Chain of questions and commands that execute after form is retrieved.
    try:
        if a.cmd == "search":
            search(a.query)
        elif a.cmd == "list":
            list_files(a.doi)
        elif a.file_id:
            download(a.file_id, None, a.format == "original")
        elif a.match and a.doi:
            hits = [
                f["dataFile"]
                for f in get_files(a.doi)
                if a.match.lower() in f["dataFile"]["filename"].lower()
            ]
            if not hits:
                sys.exit(f"No file names contain '{a.match}'. Run the list command to see them.")
            for df in hits:
                download(df["id"], df["filename"], a.format == "original")
        else:
            sys.exit("Use --file-id <number>, or give a DOI together with --match <text>.")
    except requests.HTTPError as e:
        sys.exit(
            f"HTTP error: {e}\nIf this persists, download the file manually from the Dataverse "
            "page (the download button) and put it in data/."
        )
    except requests.RequestException as e:
        sys.exit(f"Network problem: {e}")


# Only run this file if it is called directly. This means that if this file is imported into another script, then functions will be defined but won't actually run
# Running main() will call all other functions defined.
if __name__ == "__main__":
    main()
