## Download the list of 2026 U.S. Senate candidates from Wikipedia
## Each row represents one race

## Required Packages
import re
from io import StringIO
import pandas as pd 
import requests
import lxml

URL = "https://en.wikipedia.org/wiki/2026_United_States_Senate_elections"
OUTPUT_FILE = "data/senate_candidates_2026.csv"

# The 35 states with a Senate race in 2026 (Florida and Ohio are special elections)
STATES = [
    "Alabama", "Alaska", "Arkansas", "Colorado", "Delaware", "Florida",
    "Georgia", "Idaho", "Illinois", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Montana", "Nebraska", "New Hampshire", "New Jersey", "New Mexico",
    "North Carolina", "Ohio", "Oklahoma", "Oregon", "Rhode Island",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Virginia",
    "West Virginia", "Wyoming",]

## Download the Wikipedia page
response = requests.get(URL, headers={"User-Agent": "forecast-project"}, timeout=30)
html = response.text

## Read the tables that have a "Candidates" column
tables = pd.read_html(StringIO(html), match="Candidates")

## Go through each row: first column = state, last column = candidates
rows = []
for table in tables:
    for state, cand_text in zip(table.iloc[:, 0], table.iloc[:, -1]):
        state = str(state)
        race_type = "special" if "Class 3" in state else "regular"
        state = state.replace("(Class 3)", "").strip()

        if state not in STATES:
            continue

        # Remove footnotes like [a], then find every "Name (Party)"
        text = re.sub(r"\[.*?\]", "", str(cand_text))
        for name, party in re.findall(r"([^()▌;]+?)\s*\(([^)]+)\)", text):
            rows.append({
                "state": state,
                "race_type": race_type,
                "name": name.strip(),
                "party": party.strip(),})
                
## Put it in a table, remove duplicates, and save
candidates = pd.DataFrame(rows).drop_duplicates()
candidates.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(candidates)} candidates in {candidates['state'].nunique()} states to {OUTPUT_FILE}")
