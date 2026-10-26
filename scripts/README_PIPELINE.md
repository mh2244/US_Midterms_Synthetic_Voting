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
- News flow: `04-collect_news.py` screens articles; `04b-summarize_news.py` samples 3 left_leaning + 3 newspapers + 3 right_leaning items per persona and concatenates the nine existing `prompt_paragraph` values, without another LLM call. `newspapers` does not mean neutral. The summaries and selected sources are saved to `data/news/persona_news_summaries.csv`. `08-build_prompts.py --news` loads each persona's summary, shared across B/C designs, and rejects unfinished or stale summaries. Summaries use headlines/feed excerpts, not full articles. Prompt IDs include a content hash; 09 only reuses current prompt/model/mode results.
- `DEM_NOT_UP` / `REP_NOT_UP` (34 / 31) are copied from your simulation script; check them.
- Raw data should be in `data/01-raw_data/` per the assignment; scripts 02/03 still write to `data/`.
- Candidate order is randomised per persona (column `left_first`); keep it, or say why not.

## Limits to write about

- 5 personas per state gives a standard error of about 22 points on a share near 50%. The test run checks the code, not the election.
- The forecast spread covers persona sampling only. Add model and repeat spread, calibration error and a shared national error before the final run (see the comments in script 11).
- Our local embedder is symmetric and small; Heath & Alexander used Gemini Embedding 2 with asymmetric embeddings.
- Design B and SSR for vote choice is an adaptation of a method tested on 0-100 ratings.

## Persona news summaries (04b)

```bash
# Preview the selection without API calls or writing output:
python scripts/04b-summarize_news.py --dry-run
# Save nine existing paragraphs per persona, then build prompts:
python scripts/04b-summarize_news.py
python scripts/08-build_prompts.py --news
```

The default run ID is today's Toronto date, or the workflow's `RUN`; use `--run YYYY-MM-DD` to reproduce an earlier day with the same inputs. Fixed seed + run ID + news pool + persona determine sampling. Each selection has nine unique URLs, three per group; different personas may share articles. If a group has fewer than three eligible articles, collection must be expanded first. Script 04b preserves each existing paragraph, numbers the items [1]-[9], and separates them with blank lines. It does not rewrite, compress, or truncate them and needs no API key. Source metadata and input fingerprints are saved in `data/news/persona_news_summaries.csv`. Script 08 rejects stale inputs or the older LLM-compressed format. Both B/C designs use the same nine-item context for a persona. Script 04 still uses an LLM to screen and summarize individual articles; script 09 still queries voting models. Longer nine-item prompts increase those voting models' input tokens compared with a compressed summary.

## Run-specific output paths

With `RUN` set (as in the Forecast run workflow), scripts 08-15 use:

- `data/02-analysis_data/run_<RUN>/` for prompts, responses and scores.
- `data/03-results/run_<RUN>/` for forecasts, comparisons and win probabilities.
- `docs/figures/run_<RUN>/` for both forecast figures.

Candidates and personas remain shared inputs under `data/02-analysis_data/`; news inputs remain under `data/news/`. `RUN=nonews` isolates baseline results in `run_nonews`. Without `RUN`, local scripts keep their original output paths. Use the same `RUN` for every stage, for example `RUN=2026-10-27 python scripts/08-build_prompts.py --news` followed by `RUN=2026-10-27 python scripts/09-query_models.py`. A news run's 04b summary must carry the same run ID. Previous files are not moved automatically. The workflow uploads run files as an artifact before committing, providing a fallback if pushing fails. Input freezing and voting-model configuration are separate from this path change.
