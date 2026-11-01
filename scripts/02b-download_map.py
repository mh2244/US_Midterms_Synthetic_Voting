#### Preamble ####
# Purpose: Downloads the outlines of the US states (GeoJSON, longitude and latitude) that the
#          Senate map in scripts/13b-make_map.py is drawn from.
# Author: Kexin (Coco) Liu
# Date: 26 October 2026
# License: MIT
# Pre-requisites: Internet access. Run once; the file is kept in the repository.
# Any other information needed? Run from the project root:  python scripts/02b-download_map.py
# Output: data/01-raw_data/us-states.json (52 features: 50 states, DC and Puerto Rico; simplified)
# Source: PublicaMundi MappingAPI, data/geojson/us-states.json on GitHub (state outlines derived
#   from US Census Bureau cartographic boundary files).

#### Workspace setup ####
import json
from pathlib import Path

import requests

URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
OUT = Path("data/01-raw_data/us-states.json")

#### Download and check ####
response = requests.get(URL, timeout=60)
response.raise_for_status()
data = response.json()
names = {f["properties"]["name"] for f in data["features"]}
assert len(names) >= 51 and {"Alaska", "Hawaii", "Maine"} <= names, "Unexpected state file"

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data))
print(f"Saved {len(names)} outlines to {OUT}")
