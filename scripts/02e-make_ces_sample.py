#### Preamble ####
# Purpose: Writes a small sample of the raw 2024 CES file, with the same file name and every
#          column of the original, so that script 03 can be run without the full download.
# Author: Kexin (Coco) Liu
# Date: 1 November 2026
# License: MIT
# Pre-requisites: The full raw file from scripts/02-download_ces.py, saved as
#   data/01-raw_data/ces/CCES24_Common_OUTPUT_vv_topost_final.csv
# Any other information needed? Run from the project root:  python scripts/02e-make_ces_sample.py
# Outputs: data/01-raw_data/ces_sample/CCES24_Common_OUTPUT_vv_topost_final.csv
# WHY a sample: the full file is 184 MB, above GitHub's 100 MB limit for one file, so it is not
#   in the repository. The CES 2024 Common Content is released under CC0 1.0 (public domain) on
#   Harvard Dataverse (doi:10.7910/DVN/X11EP6), so a sample of real rows can be shared.
# HOW: 20 respondents from every state (inputstate), drawn at random with the project seed, so
#   every Senate state has eligible validated voters for scripts 03 and 07. Rows and columns are
#   copied unchanged. The sample is for checking that the code runs, not for any estimate.

#### Workspace setup ####
from pathlib import Path

import pandas as pd

import config

PER_STATE = 20
OUT = config.RAW / "ces_sample" / config.CES_RAW.name

#### Sample ####
raw = pd.read_csv(config.CES_RAW, low_memory=False, dtype=str, keep_default_na=False)
sample = raw.groupby("inputstate").sample(n=PER_STATE, random_state=config.SEED).sort_index()

#### Checks and save ####
assert list(sample.columns) == list(raw.columns), "The sample must keep every raw column, in order"
assert sample["inputstate"].nunique() == raw["inputstate"].nunique(), "A state is missing"
OUT.parent.mkdir(parents=True, exist_ok=True)
sample.to_csv(OUT, index=False)
print(f"Saved {len(sample):,} of {len(raw):,} rows x {sample.shape[1]} columns to {OUT}")
