# U.S. Midterm Synthetic Voting

Research project using survey, polling, candidate, and news data to study
synthetic voter responses in the 2026 U.S. Senate elections.

## Data

- **CES 2024:** demographics, validated-voter weights, and Senate/presidential vote responses from [Harvard Dataverse](https://dataverse.harvard.edu/).
- **Senate polls:** 2026 general-election polling from [PollResults.org](https://pollresults.org/api/), which uses New York Times polling data.
- **News:** RSS headlines/excerpts from ten publishers, supplemented by Google News, saved to `data/news/news.csv`. Publisher groups follow [paper section 2.2](paper.pdf#page=4).
- **Candidates:** 2026 Senate candidate list collected from [Wikipedia](https://en.wikipedia.org/wiki/2026_United_States_Senate_elections) in `data/senate_candidates_2026.csv`.

## News collection

One script: RSS collection, then OpenRouter filtering and prompt paragraphs.
Set `OPENROUTER_API_KEY` in the project-root `.env` before running:

```bash
python -m pip install -r requirements.txt
python scripts/04-collect_news.py
```

The script reads publisher RSS/Atom feeds first and supplements sparse/failed
feeds with a Google News `site:domain` query. It checks publisher domains and
RSS publication timestamps, removes duplicate URLs and repeated titles within a
publisher, and applies the article limit separately to each publisher.
Only news published in the **48 hours before the run starts** is eligible, and
**at most 60 articles per publisher** are saved, with no global cap. Articles are sorted newest first.
Feeds may contain more candidate records; no linked article pages are downloaded.
Missing, timezone-free or updated-only dates are skipped rather than guessed.

| Group (`diet`) | Publishers |
|---|---|
| `newspapers` | The New York Times, Washington Post, Wall Street Journal, USA Today |
| `right_leaning` | Fox News, New York Post, The Daily Wire |
| `left_leaning` | HuffPost, Mother Jones, Vox |

These are paper-defined source groups, not article-level bias ratings. Newspapers
are not automatically labeled neutral. This is recent article collection, not a
replication of the paper's archived-homepage summaries. Wikipedia and no-news
experimental inputs are not collected as media articles by this script.

Each run replaces **`data/news/news.csv`** with the eligible selected items:

- `published_at_utc`, `collected_at_utc`: publication and run-start timestamps.
- `outlet`, `diet`: publisher and source group.
- `title`, `summary`: headline and at most 1,000 characters of publisher RSS excerpt.
- `url`, `feed_url`: news link and discovery feed; Google links may redirect to the publisher.
- `acquisition_method`: `rss` or `google_news`.
- `text_basis`: `feed_excerpt` or `headline_only`; neither means full article text.

Google descriptions often repeat the headline, so their `summary` is left empty.
This script does not download full bodies, independently verify facts, or use
Archive. LLM filtering and paragraphs use only the supplied titles and excerpts. A publisher may contribute fewer than 60 items. Source errors
are logged, and the remaining sources continue. If every feed fails, the script
exits with an error and preserves the previous output; that file must not be
mistaken for a successful fresh run. If valid feeds return no in-window items,
the output contains only column headers. Existing legacy datasets/cache files
are left alone.

To lower the per-publisher limit or choose another output file:

```bash
python scripts/04-collect_news.py --per-source-limit 30 --out /tmp/news.csv
```

`--per-source-limit` (also accepted as `--limit`) accepts 1–60 per publisher; the 48-hour window is fixed. Requests have connection/read
timeouts, the terminal shows progress by outlet, and CSV replacement is atomic.
Dependencies include `requests`, `tqdm`, and `python-dotenv`.

## LLM filtering and prompt paragraphs

The default free model is `dots-studio/dots-3-note-preview:free`. Override with
`--model` or `OPENROUTER_NEWS_MODEL` in `.env`. The script verifies that the model
is listed, has zero prompt/completion prices, and permits disabling reasoning.
Paid models and automatic paid fallbacks are not enabled. Environment values take
precedence over `.env`.

Keep an item only if it has both US national significance and substantive relevance
to election background: federal law, policy, courts, economic developments, jobs,
inflation, healthcare, immigration, taxes, energy, education, voting rules or
national election developments. Explicitly mentioning an election is not required.
Purely local/state stories, sports, entertainment, personal gossip and unsupported
opinion are excluded. Mentioning a politician or a political controversy alone
is insufficient. A local ICE incident or a sporting boycott does not automatically
qualify as national policy. Kept items must use a supported substantive topic.

A kept item gets one neutral English `prompt_paragraph`, generally 30–80 words.
Each paragraph is prefixed with the reporting outlet. Headline-only evidence
permits a shorter attributed sentence; the model must not
pad with external knowledge or invented details. Allegations/uncertainty must be
attributed. This is not independent fact-checking and does not establish that a
story causes changes in voting behavior. It does not merge differently titled
reports into unique events.

Outputs next to the raw `news.csv`:

- `news_context.csv`: only kept news, including `prompt_paragraph`, topic,
  publisher, timestamp, evidence basis and source URL. Use `prompt_paragraph`
  as the news input to the simulation, with provenance available in the same row.
- `news_screening.csv`: every input row with `llm_status` (`keep`, `exclude`,
  `error`, `not_processed`), national/relevance flags, topic and reason. Errors
  and skipped rows are not treated as irrelevant news.

To filter the existing raw CSV without repeating collection:

```bash
python scripts/04-collect_news.py --filter-only
```

This screens the saved collection snapshot; it does not refresh the publication
window. To collect without a key or model calls:

```bash
python scripts/04-collect_news.py --collect-only
```

Requests process 20 articles by default (`--batch-size 1` through `20`). Valid
completed decisions are cached per model, prompt and article content in the
Git-ignored `.cache/news_llm.json`. Re-running `--filter-only` reuses them and
retries errors/unprocessed rows. A 429 stops new batches; completed results are
saved. Invalid JSON/decisions are retried once. Batches that remain malformed or encounter
a provider content filter are subdivided to isolate unprocessed items. A singly
blocked item remains an error; it is not marked irrelevant or sent to another
provider to bypass the filter. Generic provider HTTP 400 also triggers bounded
subdivision, stopping at five articles to limit request consumption. Error messages
include available provider details. Free daily/provider limits still
apply; the script cannot guarantee all 600 possible inputs finish in one run.
Never interpret an empty/partial final CSV after an API failure as no relevant news.

## GitHub Actions

Polling and news workflows run daily and can also be started manually from the
Actions tab. News runs `scripts/04-collect_news.py` and uploads `data/news/*.csv`
as artifacts. Set the repository secret `OPENROUTER_API_KEY` to enable screening.
