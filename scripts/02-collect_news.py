"""Collect Senate news metadata; save CSV only. Python 3.10+, no packages/key.
Run: python3 scripts/02-collect_news_google.py
Test: python3 scripts/02-collect_news_google.py --states AK
Resume: repeat the same command. New window: add --new-run.
Results: news_google/current/news_leads.csv and collection_status.csv.
Saves at most 20 newest feed results per state; shortlists 5 by default.
RSS search is not a guaranteed complete archive. Links may be Google redirects.
Publication dates are feed-reported, not independently verified.
Metadata are search leads, not verified summaries or full article text.
"""
import argparse
import csv
import hashlib
import xml.etree.ElementTree as ET
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

STATES = dict(x.split(':') for x in (
    'AL:Alabama|AK:Alaska|AR:Arkansas|CO:Colorado|DE:Delaware|FL:Florida|'
    'GA:Georgia|ID:Idaho|IL:Illinois|IA:Iowa|KS:Kansas|KY:Kentucky|LA:Louisiana|'
    'ME:Maine|MA:Massachusetts|MI:Michigan|MN:Minnesota|MS:Mississippi|MT:Montana|'
    'NE:Nebraska|NH:New Hampshire|NJ:New Jersey|NM:New Mexico|NC:North Carolina|'
    'OH:Ohio|OK:Oklahoma|OR:Oregon|RI:Rhode Island|SC:South Carolina|'
    'SD:South Dakota|TN:Tennessee|TX:Texas|VA:Virginia|WV:West Virginia|WY:Wyoming'
).split('|'))
API = 'https://news.google.com/rss/search'
ROOT = Path(__file__).resolve().parents[1] / 'news_google'
FIELDS = ['article_id', 'state', 'title', 'source_domain', 'url', 'seen_at_utc',
          'published_at', 'retrieved_at_utc', 'topic_hint', 'title_cluster_id',
          'shortlisted', 'review_status', 'filter_notes', 'summary']
STATUS_FIELDS = ['state', 'status', 'count', 'window_start_utc', 'window_end_utc',
                 'sources', 'max_per_state_shortlisted', 'error']


def parse_date(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Datetime must include timezone, e.g. +00:00.')
    return result.astimezone(timezone.utc)


def save_csv(path, rows, fields):
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
    temp.replace(path)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def canonical_url(url):
    p = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if not k.lower().startswith('utm_') and k.lower() not in ('fbclid', 'gclid')]
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path, urlencode(query), ''))


def topic(title):
    for label, pattern in {
        'debate': r'debate', 'endorsement': r'endors|backing',
        'policy': r'policy|health care|abortion|immigration|tax|economy|tariff',
        'fundraising': r'fundrais|donor|spending',
        'candidate_change': r'withdraw|drop.out',
        'campaign_event': r'rally|campaign stop|town hall',
    }.items():
        if re.search(pattern, title, re.I):
            return label
    return 'other'


def retry_after(value):
    if not value:
        return 0
    try:
        return max(0, float(value))
    except ValueError:
        try:
            return max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return 0


class Client:
    def __init__(self, interval, retries):
        self.interval, self.retries, self.last = interval, retries, None

    def get(self, url):
        for attempt in range(self.retries + 1):
            if self.last is not None:
                time.sleep(max(0, self.interval - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            server_wait = 0
            try:
                request = Request(url, headers={'User-Agent': 'SenateNewsResearch/1.0'})
                with urlopen(request, timeout=60) as response:
                    body = response.read()
                data = ET.fromstring(body)
                if data.tag != 'rss' or data.find('channel') is None:
                    raise ValueError('Unexpected RSS response; not counted as zero articles.')
                return data.findall('./channel/item')
            except HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f'HTTP {e.code}: {e.reason}') from e
                reason = f'HTTP {e.code}'
                server_wait = retry_after(e.headers.get('Retry-After'))
                e.close()
            except (URLError, TimeoutError, OSError) as e:
                reason = str(e)
            except (ValueError, UnicodeError, ET.ParseError) as e:
                reason = f'Invalid API response: {e}'
            if attempt == self.retries:
                raise RuntimeError(f'{reason}; retries exhausted.')
            delay = max(server_wait, self.interval, 30 * 2 ** attempt)
            print(f'{reason}; retry in {delay:.0f}s ({attempt + 1}/{self.retries}).', flush=True)
            time.sleep(delay)


def collect(code, start, end, sources, client):
    # Date operators narrow the search; exact UTC filtering happens below.
    query = (f'"{STATES[code]}" ("U.S. Senate" OR "US Senate" OR "Senate race") '
             f'after:{(start - timedelta(days=1)).date()} before:{(end + timedelta(days=1)).date()}')
    if sources:
        query += ' (' + ' OR '.join('site:' + source for source in sources) + ')'
    params = dict(q=query, hl='en-US', gl='US', ceid='US:en')
    articles = client.get(API + '?' + urlencode(params))
    rows, seen_urls = [], set()
    retrieved = datetime.now(timezone.utc).isoformat()
    for article in articles:
        url = (article.findtext('link') or '').strip()
        title = (article.findtext('title') or '').strip()
        if not url.startswith(('http://', 'https://')) or not title or url in seen_urls:
            continue
        try:
            published = parsedate_to_datetime(article.findtext('pubDate') or '')
            if published.tzinfo is None:
                continue
            published = published.astimezone(timezone.utc)
        except (ValueError, TypeError, OverflowError):
            continue
        if not start <= published <= end:
            continue
        seen_urls.add(url)
        source = article.find('source')
        source_url = source.get('url', '') if source is not None else ''
        normalized = re.sub(r'[^a-z0-9]+', ' ', title.lower()).strip()
        rows.append(dict(article_id=hashlib.sha256(url.encode()).hexdigest()[:20],
                         state=code, title=title,
                         source_domain=urlsplit(source_url).netloc.lower(),
                         url=url, seen_at_utc='', published_at=published.isoformat(),
                         retrieved_at_utc=retrieved, topic_hint=topic(title),
                         title_cluster_id=hashlib.sha256(normalized.encode()).hexdigest()[:16],
                         shortlisted=False, review_status='needs_review',
                         filter_notes='google_news_redirect_url;feed_publication_time_unverified', summary=''))
    rows.sort(key=lambda r: r['published_at'], reverse=True)
    return rows[:20]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--states', nargs='+')
    p.add_argument('--days', type=int, default=7)
    p.add_argument('--max-per-state', type=int, default=5, help='Number to shortlist, not download limit.')
    p.add_argument('--until')
    p.add_argument('--sources', default='')
    p.add_argument('--interval', type=float, default=15, help='Minimum seconds between requests (default 15).')
    p.add_argument('--retries', type=int, default=4)
    p.add_argument('--new-run', action='store_true', help='Archive current results and start a fresh window.')
    args = p.parse_args()
    states = [s.upper() for s in (args.states or STATES)]
    if len(states) != len(set(states)) or set(states) - STATES.keys():
        p.error('Unknown or duplicate state codes.')
    if args.days < 1 or args.max_per_state < 1 or args.interval < 5 or args.retries < 0:
        p.error('days/max-per-state must be positive; interval >= 5; retries >= 0.')
    sources = sorted(set(s.strip().lower() for s in args.sources.split(',') if s.strip()))
    if any(not re.fullmatch(r'[a-z0-9.-]+', s) for s in sources):
        p.error('Invalid source domain.')
    run_dir = ROOT / 'current'
    if args.new_run and run_dir.exists():
        archive = ROOT / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        run_dir.rename(archive)
        print(f'Previous run archived: {archive}')
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path, status_path = run_dir / 'news_leads.csv', run_dir / 'collection_status.csv'
    statuses = read_csv(status_path)
    rows = read_csv(csv_path)
    if statuses:
        if [s['state'] for s in statuses] != states:
            p.error('Resume requires same --states; use --new-run for different states.')
        first = statuses[0]
        start, end = parse_date(first['window_start_utc']), parse_date(first['window_end_utc'])
        if (first['sources'] != ','.join(sources)
                or int(first['max_per_state_shortlisted']) != args.max_per_state
                or end - start != timedelta(days=args.days)
                or (args.until and parse_date(args.until) != end)):
            p.error('Resume options differ; use original options or --new-run.')
        print(f'Resuming fixed window: {start.isoformat()} to {end.isoformat()}')
    else:
        if rows:
            p.error('Missing collection_status.csv; preserve this folder and use --new-run.')
        end = parse_date(args.until) if args.until else datetime.now(timezone.utc)
        start = end - timedelta(days=args.days)
        statuses = [dict(state=s, status='pending', count=0, window_start_utc=start.isoformat(),
                         window_end_utc=end.isoformat(), sources=','.join(sources),
                         max_per_state_shortlisted=args.max_per_state, error='') for s in states]
        save_csv(status_path, statuses, STATUS_FIELDS)
    client = Client(args.interval, args.retries)
    for item in statuses:
        code = item['state']
        if item['status'] == 'completed':
            print(f'{code}: skipped (already completed).', flush=True)
            continue
        try:
            state_rows = collect(code, start, end, sources, client)
        except (RuntimeError, ValueError) as e:
            item.update(status='failed', error=str(e))
            save_csv(status_path, statuses, STATUS_FIELDS)
            print(f'{code}: stopped: {e}\nSaved results remain at {csv_path}\nRepeat the same command to resume.', flush=True)
            return 1
        for row in state_rows[:args.max_per_state]:
            row['shortlisted'] = True
        # Replace this state's rows: safe if interrupted between the two saves.
        rows = [r for r in rows if r['state'] != code] + state_rows
        save_csv(csv_path, rows, FIELDS)
        item.update(status='completed', count=len(state_rows), error='')
        save_csv(status_path, statuses, STATUS_FIELDS)
        print(f'{code}: collected {len(state_rows)} articles; CSV saved.', flush=True)
    print(f'Done: {len(rows)} article rows. Output: {csv_path}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit('\nInterrupted. Repeat the same command to resume.') from None
    except (OSError, ValueError) as error:
        raise SystemExit(f'Collection stopped: {error}') from None
