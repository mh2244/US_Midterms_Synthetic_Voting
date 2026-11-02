# SAMPLE of the raw 2024 CES file — not the full data

`CCES24_Common_OUTPUT_vv_topost_final.csv` here has the same file name and all 694 columns of the
2024 Cooperative Election Study Common Content (vote-validated file), but only 1,020 of its 60,000
respondents: 20 drawn at random from each state by `scripts/02e-make_ces_sample.py` (seed 2026).
Rows and values are copied unchanged.

- **Why a sample:** the full file is 184 MB, above GitHub's 100 MB limit for one file. The data are
  public domain (CC0 1.0): Schaffner, Shih, Ansolabehere and Pope (2025), *Cooperative Election
  Study Common Content, 2024*, Harvard Dataverse, doi:10.7910/DVN/X11EP6.
- **Use:** checking that the code runs. It is far too small for any estimate.
- **Run the extraction on it** (writes to a separate file, so the real extract is not overwritten):

  ```bash
  python scripts/03-extract_ces.py data/01-raw_data/ces_sample/CCES24_Common_OUTPUT_vv_topost_final.csv \
    --out data/02-analysis_data/ces_sample_extract.csv
  ```

- **Full file:** `python scripts/02-download_ces.py get --file-id 12050325`, saved as
  `data/01-raw_data/ces/CCES24_Common_OUTPUT_vv_topost_final.csv`.
