# U.S. Midterm Synthetic Voting

This project uses CES survey respondents as synthetic voter personas to study the
2026 U.S. Senate elections. Four language models answer voting questions; their
responses are scored and aggregated into simulated state outcomes.

**Current design:** 35 states × 100 personas × 4 voting models × 3 repeats =
**42,000 voting requests per run**, each a free-text answer scored with SSR. News screening and embedding
requests are additional. Model-derived scores and simulations are not verified
individual voting probabilities.

## Setup

Use Python 3.12 (the version used by GitHub Actions). From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a local `.env` file:

```dotenv
OPENROUTER_API_KEY=your_key_here
```

Do not commit `.env`. For GitHub Actions, add the same variable name as a
repository secret under **Settings → Secrets and variables → Actions**.
The default news, voting and embedding models all require OpenRouter credits.

## Data and scripts

| Stage | Scripts | What they do |
|---|---|---|
| Artificial test data | `00`, `01` | Generate and check fake personas, answers and election outcomes; no model calls |
| CES survey inputs | `02`, `03` | Download CES 2024 from [Harvard Dataverse](https://dataverse.harvard.edu/) and extract demographics, weights and vote variables |
| Polling inputs | `polling_data/00`, `polling_data/01` | Download [PollResults.org](https://pollresults.org/api/) records and retain 2026 general-election Senate polls |
| Candidates | `05`, `06` | Collect the [Wikipedia candidate list](https://en.wikipedia.org/wiki/2026_United_States_Senate_elections), select the main two sides, and flag ambiguities for review |
| Personas | `07` | Sample validated 2024 voters within each state, with replacement, proportional to `vvweight_post` |
| News | `04`, `04b` | Collect and screen articles, then assemble nine existing news paragraphs per persona |
| Prompts and responses | `08`, `09` | Build prompts and query the four voting models |
| Scoring and forecast | `10`, `11` | Apply SSR or parse percentages; simulate outcomes and compare with polling averages |
| Checks and figures | `12`, `13`, `15` | Check output consistency; draw outcome distributions and win-probability bars |
| Persona composition | `14` | Summarize the demographic composition of the persona sample |

All scripts are in [`scripts/`](scripts/). Shared inputs are
`data/ces2024_senate_demographics.csv`, `data/02-analysis_data/personas.parquet`,
and `data/02-analysis_data/senate_nominees_2026.csv`.
Prepare these before running the forecast; the forecast workflow does not rebuild
personas or candidates. Review candidate warnings rather than assuming scraped
names are correct.

## Run a news experiment locally

Set the same `RUN` for every step. The planned news runs are `2026-10-27`,
`2026-10-29`, and `2026-10-31`; the default in `config.py` is `2026-10-27`.

```bash
export RUN=2026-10-27

python scripts/polling_data/00-download_polling_data.py
python scripts/polling_data/01-clean_polling_data.py
python scripts/04-collect_news.py
python scripts/04b-summarize_news.py
python scripts/08-build_prompts.py --news
python scripts/09-query_models.py --batch-size 40 --sleep 0
python scripts/10-score_responses.py
python scripts/11-build_forecast.py
python scripts/12-test_actual_data.py
python scripts/13-make_figures.py
python scripts/15-make_summary_bar.py
```

These commands make paid model requests. For a small query pilot, use
`python scripts/09-query_models.py --limit 4`. Answers are saved, but exit code 1
is expected while the full request set remains unfinished. To continue querying,
rerun `09` with the same prompts. Rebuilding news/prompts may change their IDs and
require new answers.

For the **no-news baseline**, set `export RUN=nonews`, skip `04` and `04b`, and
run `08` without `--news`, followed by `09–15` as above (excluding optional `14`).
Each prompt asks for a short first-person voting answer in free text.

## How news is used

`04` reads RSS/Atom feeds, with Google News as a supplement. It keeps the previous
**48 hours**, up to **60 articles per publisher**, without a global cap. It uses
headlines and feed excerpts, not full article bodies.

| Source group | Publishers |
|---|---|
| `left_leaning` | HuffPost, Mother Jones, Vox |
| `newspapers` | The New York Times, Washington Post, Wall Street Journal, USA Today |
| `right_leaning` | Fox News, New York Post, The Daily Wire |

These are source groups, not article-level bias measurements; `newspapers` does
not mean neutral. The LLM retains national, election-relevant policy/economic
background and writes one attributed paragraph per item. A dated role fact list
and conflict-word rules trigger a rewrite or exclusion for inconsistent titles.
This is not independent verification of the reported events.

`04b` randomly selects **three unique articles per group for each persona** and
concatenates their nine paragraphs unchanged. It makes no model calls. Different
personas may share articles; fixed inputs, RUN and seed reproduce the selection.
Each group must have at least three eligible articles. A persona's B/C prompts
use the same selection.

Useful options:

```bash
python scripts/04-collect_news.py --collect-only  # no screening/model calls
python scripts/04-collect_news.py --filter-only   # rescreen saved news; no new collection
python scripts/04b-summarize_news.py --dry-run    # preview selections; no output writes
```

## Models

| Purpose | Default model ID |
|---|---|
| News screening, single-item summaries and corrective rewrites (`04`) | `openai/gpt-4.1-nano` |
| Voting responses (`09`) | `anthropic/claude-haiku-5.5` |
| Voting responses (`09`) | `openai/gpt-6-luna` |
| Voting responses (`09`) | `deepseek/deepseek-v4.1-flash` |
| Voting responses (`09`) | `google/gemini-3.5-flash-lite` |
| SSR embeddings (`10`) | `google/gemini-embedding-2` |

Voting, embedding and sampling settings are in [`scripts/config.py`](scripts/config.py).
The news default is in `04`; override it with `--model` or
`OPENROUTER_NEWS_MODEL`. Historical pilot models and AI-assisted development are
recorded in [`plan/LLM Usage Document.txt`](plan/LLM%20Usage%20Document.txt).

# U.S. Midterm Synthetic Voting

This project uses CES survey respondents as synthetic voter personas to study the
2026 U.S. Senate elections. Four language models answer voting questions; their
responses are scored and aggregated into simulated state outcomes.

**Current design:** 35 states × 100 personas × 4 voting models × 3 repeats =
**42,000 voting requests per run**, each a free-text answer scored with SSR. News screening and embedding
requests are additional. Model-derived scores and simulations are not verified
individual voting probabilities.

## Setup

Use Python 3.12 (the version used by GitHub Actions). From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a local `.env` file:

```dotenv
OPENROUTER_API_KEY=your_key_here
```

Do not commit `.env`. For GitHub Actions, add the same variable name as a
repository secret under **Settings → Secrets and variables → Actions**.
The default news, voting and embedding models all require OpenRouter credits.

## Data and scripts

| Stage | Scripts | What they do |
|---|---|---|
| Artificial test data | `00`, `01` | Generate and check fake personas, answers and election outcomes; no model calls |
| CES survey inputs | `02`, `03` | Download CES 2024 from [Harvard Dataverse](https://dataverse.harvard.edu/) and extract demographics, weights and vote variables |
| Polling inputs | `polling_data/00`, `polling_data/01` | Download [PollResults.org](https://pollresults.org/api/) records and retain 2026 general-election Senate polls |
| Candidates | `05`, `06` | Collect the [Wikipedia candidate list](https://en.wikipedia.org/wiki/2026_United_States_Senate_elections), select the main two sides, and flag ambiguities for review |
| Personas | `07` | Sample validated 2024 voters within each state, with replacement, proportional to `vvweight_post` |
| News | `04`, `04b` | Collect and screen articles, then assemble nine existing news paragraphs per persona |
| Prompts and responses | `08`, `09` | Build prompts and query the four voting models |
| Scoring and forecast | `10`, `11` | Apply SSR or parse percentages; simulate outcomes and compare with polling averages |
| Checks and figures | `12`, `13`, `15` | Check output consistency; draw outcome distributions and win-probability bars |
| Persona composition | `14` | Summarize the demographic composition of the persona sample |

All scripts are in [`scripts/`](scripts/). Shared inputs are
`data/ces2024_senate_demographics.csv`, `data/02-analysis_data/personas.parquet`,
and `data/02-analysis_data/senate_nominees_2026.csv`.
Prepare these before running the forecast; the forecast workflow does not rebuild
personas or candidates. Review candidate warnings rather than assuming scraped
names are correct.

## Run a news experiment locally

Set the same `RUN` for every step. The planned news runs are `2026-10-27`,
`2026-10-29`, and `2026-10-31`; the default in `config.py` is `2026-10-27`.

```bash
export RUN=2026-10-27

python scripts/polling_data/00-download_polling_data.py
python scripts/polling_data/01-clean_polling_data.py
python scripts/04-collect_news.py
python scripts/04b-summarize_news.py
python scripts/08-build_prompts.py --news
python scripts/09-query_models.py --batch-size 40 --sleep 0
python scripts/10-score_responses.py
python scripts/11-build_forecast.py
python scripts/12-test_actual_data.py
python scripts/13-make_figures.py
python scripts/15-make_summary_bar.py
```

These commands make paid model requests. For a small query pilot, use
`python scripts/09-query_models.py --limit 4`. Answers are saved, but exit code 1
is expected while the full request set remains unfinished. To continue querying,
rerun `09` with the same prompts. Rebuilding news/prompts may change their IDs and
require new answers.

For the **no-news baseline**, set `export RUN=nonews`, skip `04` and `04b`, and
run `08` without `--news`, followed by `09–15` as above (excluding optional `14`).
Each prompt asks for a short first-person voting answer in free text.

## How news is used

`04` reads RSS/Atom feeds, with Google News as a supplement. It keeps the previous
**48 hours**, up to **60 articles per publisher**, without a global cap. It uses
headlines and feed excerpts, not full article bodies.

| Source group | Publishers |
|---|---|
| `left_leaning` | HuffPost, Mother Jones, Vox |
| `newspapers` | The New York Times, Washington Post, Wall Street Journal, USA Today |
| `right_leaning` | Fox News, New York Post, The Daily Wire |

These are source groups, not article-level bias measurements; `newspapers` does
not mean neutral. The LLM retains national, election-relevant policy/economic
background and writes one attributed paragraph per item. A dated role fact list
and conflict-word rules trigger a rewrite or exclusion for inconsistent titles.
This is not independent verification of the reported events.

`04b` randomly selects **three unique articles per group for each persona** and
concatenates their nine paragraphs unchanged. It makes no model calls. Different
personas may share articles; fixed inputs, RUN and seed reproduce the selection.
Each group must have at least three eligible articles. A persona's B/C prompts
use the same selection.

Useful options:

```bash
python scripts/04-collect_news.py --collect-only  # no screening/model calls
python scripts/04-collect_news.py --filter-only   # rescreen saved news; no new collection
python scripts/04b-summarize_news.py --dry-run    # preview selections; no output writes
```

## Models

| Purpose | Default model ID |
|---|---|
| News screening, single-item summaries and corrective rewrites (`04`) | `openai/gpt-4.1-nano` |
| Voting responses (`09`) | `anthropic/claude-haiku-5.5` |
| Voting responses (`09`) | `openai/gpt-6-luna` |
| Voting responses (`09`) | `deepseek/deepseek-v4.1-flash` |
| Voting responses (`09`) | `google/gemini-3.5-flash-lite` |
| SSR embeddings (`10`) | `google/gemini-embedding-2` |

Voting, embedding and sampling settings are in [`scripts/config.py`](scripts/config.py).
The news default is in `04`; override it with `--model` or
`OPENROUTER_NEWS_MODEL`. Historical pilot models and AI-assisted development are
recorded in [`plan/LLM Usage Document.txt`](plan/LLM%20Usage%20Document.txt).

## Outputs

| Location | Contents |
|---|---|
| `data/00-simulated_data/` | Artificial test data |
| `data/polling_data/` | Timestamped raw polling downloads and cleaned poll answers |
| `data/news/` | Raw news, screening decisions, kept paragraphs and per-persona selections |
| `data/02-analysis_data/run_<RUN>/` | Prompts, raw model responses, scores and readable scored answers |
| `data/03-results/run_<RUN>/` | Forecast draws, polling averages, state comparisons, chamber summary and win probabilities |
| `docs/figures/run_<RUN>/` | Distribution plot and state win-probability bar chart |

`news_screening.csv` includes exclusions, errors and role-check audit fields;
`news_context.csv` contains kept articles only. Per-persona selections are in
`persona_news_summaries.csv`. The forecast workflow archives news with date-suffixed
filenames and saves the polling input as `run_<RUN>/polling_input.csv`.

Forecast shares are normalized between the two selected candidates. Polling
averages use sample-size weights within a 45-day window ending at the newest
poll in the file. Polls are a comparison by default; `11 --calibrate` enables an
optional correction. Simulations include persona resampling and an assumed
national error (default SD 0.03); this is not a complete election-error model.

## GitHub Actions and website

- **Forecast run:** scheduled for October 27, 29 and 31 at 18:13 UTC (2:13 p.m.
  Toronto on those dates). It refreshes polling data, runs the news/forecast
  pipeline and uploads artifacts. It saves news, polling inputs, answers and
  scores even after a failure; forecasts and figures are committed only when
  all preceding steps succeed. `09` exits nonzero when queries are incomplete.
- **Manual forecast:** Actions → Forecast run → Run workflow. Enter a date or
  `nonews`; leave blank for today's Toronto date. A rerun recollects news; it is
  not a frozen-input resume.
- **Polling data:** a separate daily download/clean workflow that uploads an
  artifact. The forecast refreshes its own polling inputs and does not depend
  on this artifact.

Schedules run from the default branch and can be delayed. The October cron repeats
each year; disable it after the project. Artifact upload provides a backup if an
automatic push is blocked. Inspect failed runs before treating their outputs as
valid results. Six-hour completion is not guaranteed for 42,000 requests.

The website source is `index.qmd`, with `_quarto.yml` and `styles.css`; rendered
files are in `docs/`. Use `quarto render` to rebuild it. The forecast workflow
creates figures but does not render or publish the website.

## Statement on AI usage

We used AI in two separate ways: to help write the code and the website
(development), and as part of the forecast itself (pipeline). Commits made with a
coding agent carry a `Co-Authored-By` line naming the model, so
`git log --grep "Co-Authored-By"` lists them.

<!-- TODO before submitting: replace every [FILL IN ...] and delete this comment. -->

### Development

**Claude Opus 5.5 (`claude-opus-5-5`), through Claude Code, a coding agent.**
Kexin (Coco) Liu used one Claude Code session from 26 to 28 October 2026 for 26
commits on `main`. In that session the model wrote or edited:

- **Pipeline scripts:** `00`, `01`, `01b`, `02b`, `04`, `04b`, `08`, `09`, `10`,
  `11`, `12`, `12b`, `13`, `13b`, `14`, `15`, `15b`, `15c`, `16`, `16b`, `16c`,
  `16d`, `17`, `config.py` and `schemas.py`. The main changes were:
  - making queries and SSR scoring robust to bad responses;
  - switching SSR to Gemini Embedding 2;
  - screening news for outdated facts;
  - the inverse-variance combination with polls;
  - the run comparison, sensitivity, cost and leave-one-out tables;
  - rewriting both test suites with pytest and pandera.
- **The GitHub Actions workflow:** `.github/workflows/news-data.yml`, which handles
  automatic runs, backup schedules, the credit check and resuming a run.
- **The website's structure and design files:** `index.qmd` layout, `_quarto.yml`,
  `styles.css`, `scripts/site.mplstyle` and the bundled fonts.
- **Documentation:** `README.md`, `scripts/README_PIPELINE.md` and `references.bib`.

Paragraphs in `index.qmd` marked `DRAFT (Claude)` were drafted by this model.
[FILL IN: whether and by whom these were rewritten.] [FILL IN: how the code was
checked before committing, e.g. "Coco read each change and ran the tests".]

**Claude Opus 5.5 (`claude-opus-5-5`), through Claude on claude.ai (an agentic
session built on Claude Code).** Maggie used it on 28 October 2026 to
review the repository against the rubric and to draft this statement. It did not
change any script.

**Claude Sonnet 5.5 (`[FILL IN: identifier]`), through [FILL IN: claude.ai chat or
Claude Code].** [FILL IN: who, and dates.] Used to clarify the analysis plan,
write [FILL IN: which scripts, e.g. the first versions of `00` and `01`], make
figures, design the website and troubleshoot errors. It also corrected minor
grammar in drafts we had written ourselves.

**Gemini 3.5 Flash Lite (`[FILL IN: identifier]`), through [FILL IN: tool].**
[FILL IN: who, and dates.] Used to clarify the analysis plan, explain concepts
from papers and troubleshoot errors. It did not write code or prose in the repo.
[FILL IN: correct this if it did.]

**[FILL IN: model and identifier], through OpenAI Codex / ChatGPT.** [FILL IN:
who, and dates.] Used for code review and edits, GitHub Actions and environment
setup, pipeline documentation (`README.md`, `scripts/README_PIPELINE.md`) and
explanations of the analysis.

[FILL IN: one sentence on prose, e.g. "Apart from the above, we wrote the
website text ourselves."]

### Pipeline

All pipeline models are called through OpenRouter. The model ID each provider
actually returned is saved with every answer in the `model_returned` column of
`data/02-analysis_data/run_<RUN>/responses.parquet`.

| Model ID (requested) | ID returned | Script | Runs | Used for |
|---|---|---|---|---|
| `openai/gpt-4.1-nano` | [FILL IN] | `04` | 26 Oct onward | Screening news items, writing one-paragraph summaries, rewriting flagged paragraphs |
| `anthropic/claude-haiku-5.5` | [FILL IN] | `09` | 26 Oct (test), 27 Oct, nonews, [FILL IN later runs] | Voting answers |
| `openai/gpt-6-luna` | [FILL IN] | `09` | as above | Voting answers |
| `deepseek/deepseek-v4.1-flash` | [FILL IN] | `09` | as above | Voting answers |
| `google/gemini-3.5-flash-lite` | [FILL IN] | `09` | as above | Voting answers |
| `google/gemini-embedding-2` | [FILL IN] | `10` | 26 Oct onward | SSR embeddings of answers and anchor statements |

Earlier test runs (`test_run_2026-10-21`, `pilot_2026-10-25`) used the free models
`poolside/laguna-s-2.1:free` and `nvidia/nemotron-3-ultra-550b-a55b:free` for voting
answers, `dots-studio/dots-3-note-preview:free` for news screening and summaries,
and the local `sentence-transformers/all-MiniLM-L6-v2` for SSR embeddings. These runs
were only for testing the pipeline.

## Outputs

| Location | Contents |
|---|---|
| `data/00-simulated_data/` | Artificial test data |
| `data/polling_data/` | Timestamped raw polling downloads and cleaned poll answers |
| `data/news/` | Raw news, screening decisions, kept paragraphs and per-persona selections |
| `data/02-analysis_data/run_<RUN>/` | Prompts, raw model responses, scores and readable scored answers |
| `data/03-results/run_<RUN>/` | Forecast draws, polling averages, state comparisons, chamber summary and win probabilities |
| `docs/figures/run_<RUN>/` | Distribution plot and state win-probability bar chart |

`news_screening.csv` includes exclusions, errors and role-check audit fields;
`news_context.csv` contains kept articles only. Per-persona selections are in
`persona_news_summaries.csv`. The forecast workflow archives news with date-suffixed
filenames and saves the polling input as `run_<RUN>/polling_input.csv`.

Forecast shares are normalized between the two selected candidates. Polling
averages use sample-size weights within a 45-day window ending at the newest
poll in the file. Polls are a comparison by default; `11 --calibrate` enables an
optional correction. Simulations include persona resampling and an assumed
national error (default SD 0.03); this is not a complete election-error model.

## GitHub Actions and website

- **Forecast run:** scheduled for October 27, 29 and 31 at 18:13 UTC (2:13 p.m.
  Toronto on those dates). It refreshes polling data, runs the news/forecast
  pipeline and uploads artifacts. It saves news, polling inputs, answers and
  scores even after a failure; forecasts and figures are committed only when
  all preceding steps succeed. `09` exits nonzero when queries are incomplete.
- **Manual forecast:** Actions → Forecast run → Run workflow. Enter a date or
  `nonews`; leave blank for today's Toronto date. A rerun recollects news; it is
  not a frozen-input resume.
- **Polling data:** a separate daily download/clean workflow that uploads an
  artifact. The forecast refreshes its own polling inputs and does not depend
  on this artifact.

Schedules run from the default branch and can be delayed. The October cron repeats
each year; disable it after the project. Artifact upload provides a backup if an
automatic push is blocked. Inspect failed runs before treating their outputs as
valid results. Six-hour completion is not guaranteed for 42,000 requests.

The website source is `index.qmd`, with `_quarto.yml` and `styles.css`; rendered
files are in `docs/`. Use `quarto render` to rebuild it. The forecast workflow
creates figures but does not render or publish the website.
