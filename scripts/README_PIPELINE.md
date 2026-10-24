# Pipeline scripts 06-13 (test run: 5 personas per state, 2 free models)

Copy the files in `scripts/` into your project's `scripts/` folder (keep `config.py` beside the others). Run everything from the project root.

## One-time setup

```
pip install pandas numpy pyarrow requests matplotlib sentence-transformers
```

Make a file called `.env` in the project root with ONE line (no quotes):

```
OPENROUTER_API_KEY=sk-or-...your new key...
```

Add `.env` to `.gitignore`. The key you pasted in chat is exposed: revoke it at openrouter.ai/keys and make a new one.

## Order

| Script | What it does | Writes |
|---|---|---|
| `06-clean_candidates.py` | one left and one Republican name per race; prints warnings to fix | `senate_nominees_2026.csv` |
| `07-build_personas.py` | draws 5 real CES respondents per state by weight; writes persona text | `personas.parquet/.csv` |
| `08-build_prompts.py` | persona + question (`--designs B,C`, `--news`) | `prompts.parquet` |
| `09-query_models.py` | asks both models (`--limit 4` first; `--mock` for a dry run) | `responses.parquet` |
| `10-score_responses.py` | SSR (design B) or parsing (design C) | `scores.parquet`, `scores_with_text.csv` |
| `11-build_forecast.py` | state shares, 1,000 draws, poll and 2024 comparison | `forecast.parquet`, `state_comparison.csv` |
| `12-test_actual_data.py` | tests on real files | prints "All tests passed" |
| `13-make_figures.py` | quantile dotplot | `docs/figures/forecast_dotplot.png` |

Outputs go to `data/02-analysis_data/` and `data/03-results/`. `config.py` holds the settings (models, N per state, paths).

## Free-model limits

20 requests per minute, and 50 per day unless you have bought 10 credits (then 1,000 per day). 175 personas x 2 models = 350 calls. Script 09 saves as it goes, skips finished prompts, retries failed ones, and stops cleanly at the daily limit. Run it again the next day.

## Not verified by me (checked offline with a fake model only)

- Live OpenRouter calls: the two model ids (`stealth/space-bunny-alpha`, `nvidia/nemotron-3-super-120b-a12b:free`), how reasoning models report empty answers, and the daily-limit message text.
- The real embedding model in script 10 (`sentence-transformers`). The fake embedder only checks the code path.
- Parquet writing (my sandbox had no pyarrow; the code falls back to csv only when pyarrow is missing).

## Fix before the final run

- Candidate file (script 06 warnings): Idaho, Nebraska and South Dakota have no Democrat listed; Idaho's polls show David Roth as Democrat; Alaska lists 3 Republicans. Edit `OVERRIDES` after checking Ballotpedia.
- News headlines are weak (many have nothing to do with the Senate). Skip `--news` for the first runs.
- `DEM_NOT_UP` / `REP_NOT_UP` (34 / 31) are copied from your simulation script; check them.
- Raw data should be in `data/01-raw_data/` per the assignment; scripts 02/03 still write to `data/`.
- Candidate order is randomised per persona (column `left_first`); keep it, or say why not.

## Limits to write about

- 5 personas per state gives a standard error of about 22 points on a share near 50%. The test run checks the code, not the election.
- The forecast spread covers persona sampling only. Add model and repeat spread, calibration error and a shared national error before the final run (see the comments in script 11).
- Our local embedder is symmetric and small; Heath & Alexander used Gemini Embedding 2 with asymmetric embeddings.
- Design B and SSR for vote choice is an adaptation of a method tested on 0-100 ratings.
