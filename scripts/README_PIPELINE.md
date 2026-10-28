# Running the forecast pipeline

See the [project README](../README.md) for setup, data sources, model IDs and
output locations. Run commands from the project root with `.venv` activated.
Settings are in [`config.py`](config.py).

## News run

Shared CES, candidate and persona files must already exist. Set the same RUN for
every stage:

```bash
export RUN=2026-10-27
python scripts/polling_data/00-download_polling_data.py
python scripts/polling_data/01-clean_polling_data.py
python scripts/04-collect_news.py
python scripts/04b-summarize_news.py
python scripts/08-build_prompts.py --news
python scripts/09-query_models.py --batch-size 40 --sleep 0
python scripts/10-score_responses.py
python scripts/11-build_forecast.py --combine none
python scripts/11-build_forecast.py --combine shrink
python scripts/11-build_forecast.py              # inverse-variance headline, run last
python scripts/13-make_figures.py
python scripts/13b-make_map.py
python scripts/15-make_summary_bar.py
python scripts/15b-make_headline.py
python scripts/16-make_crosstabs.py
python scripts/16b-compare_runs.py
python scripts/16c-sensitivity.py
python scripts/12-test_actual_data.py
python scripts/17-publish_latest.py              # news runs only; then quarto render
```

Planned news dates are October 27, 29 and 31, 2026. Without an override, RUN is
`2026-10-27`. For the baseline, use `export RUN=nonews`, skip 04/04b, and run 08
without `--news`, followed by the remaining stages. Optional script 14 summarizes
persona composition and is not part of the forecast workflow.

These commands make paid requests: news uses GPT-4.1 Nano, voting uses four
models, and SSR uses Gemini Embedding 2. With 3,500 personas and three repeats,
Design B requires 42,000 voting requests per run, plus screening/embedding.

## What to expect

- **04:** collect the last 48 hours, up to 60 articles per publisher; screen and
  summarize supplied headlines/excerpts. Role conflicts trigger one corrective
  rewrite or exclusion. This is not independent verification of reported facts.
- **04b:** sample three articles per source group for each persona and concatenate
  nine existing paragraphs. No extra LLM calls; at least three eligible articles
  per group are required. `newspapers` does not mean neutral.
- **08:** validate persona/news inputs and build prompts. Content hashes and RUN
  identify the prompt version. B/C share the persona's news selection.
- **09:** save raw responses and reuse matching successes. Incomplete runs exit
  with code 1, including an unfinished `--limit` pilot. Keep prompts unchanged
  and rerun 09 to retry remaining work.
- **10–16:** score, simulate, plot and test. Script 11 combines the synthetic share
  with polls (or 2024 + uniform swing) by inverse-variance weights; `--combine
  shrink` and `--combine none` are kept for comparison. 13b, 15b and 16 write the
  map, headline and crosstabs. Tests check consistency, not prediction accuracy.
- **17:** copies the latest news run to `figures/latest/` and
  `data/03-results/latest/`, which `index.qmd` reads.

Outputs are isolated under `data/02-analysis_data/run_<RUN>/`,
`data/03-results/run_<RUN>/`, and `figures/run_<RUN>/` (not `docs/`, which Quarto rebuilds). Shared candidates and
personas stay in `data/02-analysis_data/`; news stays in `data/news/`.

## Useful commands

```bash
python scripts/04-collect_news.py --collect-only  # collect without model calls
python scripts/04-collect_news.py --filter-only   # rescreen the saved collection
python scripts/04b-summarize_news.py --dry-run    # preview without writing
python scripts/09-query_models.py --limit 4       # paid pilot; partial exit code 1
```

After changing the news pool or personas, rerun 04b and 08. If using 04b's
`--run`, it must match config.RUN. Rebuilding prompts can require new voting
requests. A full GitHub workflow rerun recollects news; it is not a frozen-input
resume. Failed workflow outputs are retained for inspection, not automatically
certified as valid results.

The workflow refreshes its own polling input, archives a copy as
`data/03-results/run_<RUN>/polling_input.csv`, uploads artifacts, and attempts to
commit news, polling inputs, answers and scores even after failure. Forecasts,
figures and the rendered website (`docs/`) are committed only when all
preceding steps, including the tests, succeed.
