#!/usr/bin/env python3
"""
Pre-amble:
Collect Google News RSS metadata for national/state public-policy news.

Python 3.10+, standard library only. No API key or LLM calls.
Run: python scripts/02-collect_news.py
Test: python scripts/02-collect_news.py --states MI,PA
Default output: data/news_google/news_google.csv and coverage.csv
Saves CSV results only; raw RSS feeds and JSON metadata are not written.
RSS descriptions are feed text, not full article text or verified summaries.
"""
import argparse
import csv
import hashlib
import html
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

STATES = dict(zip(
    'AL AK AR CO DE FL GA ID IL IA KS KY LA ME MA MI MN MS MT NE NH NJ NM NC OH OK OR RI SC SD TN TX VA WV WY'.split(),
    ['Alabama', 'Alaska', 'Arkansas', 'Colorado', 'Delaware', 'Florida',
     'Georgia', 'Idaho', 'Illinois', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
     'Maine', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Montana',
     'Nebraska', 'New Hampshire', 'New Jersey', 'New Mexico', 'North Carolina',
     'Ohio', 'Oklahoma', 'Oregon', 'Rhode Island', 'South Carolina',
     'South Dakota', 'Tennessee', 'Texas', 'Virginia', 'West Virginia', 'Wyoming']))
# One combined query per scope. No topic labels or per-topic quotas.
ISSUES = '(economy OR inflation OR jobs OR healthcare OR housing OR immigration OR legislation OR "public services" OR energy OR "disaster relief")'
EXCLUDE = re.compile(r'\b(opinion|editorial|endorsement|campaign|debate|polling|polls|senate race)\b', re.I)
FIELDS = ['article_id', 'level', 'state', 'title', 'source', 'source_url',
          'published_at_utc', 'google_news_url', 'feed_description', 'review_status']

def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clean(value):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', value or ''))).strip()


def fetch(url):
    for attempt in range(3):
        try:
            request = Request(url, headers={'User-Agent': 'AcademicNewsRSSCollector/1.0'})
            with urlopen(request, timeout=30) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(3 * (attempt + 1))


def parse_feed(raw, level, state, start, cutoff):
    root = ET.fromstring(raw)
    channel = root.find('channel')
    if root.tag != 'rss' or channel is None:
        raise ValueError('Unexpected response: not a Google News RSS feed.')
    rows, audit = [], []
    for item in channel.findall('item'):
        title = clean(item.findtext('title'))
        url = (item.findtext('link') or '').strip()
        source = item.find('source')
        publisher = clean(source.text if source is not None else '')
        # Drop the publisher suffix only when it matches the source element.
        if publisher and title.endswith(' - ' + publisher):
            title = title[:-len(' - ' + publisher)]
        reason = ''
        published = None
        try:
            published = parsedate_to_datetime(item.findtext('pubDate') or '')
            if published.tzinfo is None:
                raise ValueError('Missing timezone')
            published = published.astimezone(timezone.utc)
        except (ValueError, TypeError, OverflowError):
            reason = 'invalid_publication_time'
        if not reason and not start <= published <= cutoff:
            reason = 'outside_time_window'
        if not title or not url:
            reason = 'missing_title_or_url'
        if not reason and EXCLUDE.search(title):
            reason = 'excluded_title_keyword'
        row = {'article_id': hashlib.sha256(url.encode()).hexdigest()[:20],
               'level': level, 'state': state or '', 'title': title,
               'source': publisher, 'source_url': source.get('url', '') if source is not None else '',
               'published_at_utc': published.isoformat() if published else '',
               'google_news_url': url, 'feed_description': clean(item.findtext('description')),
               'review_status': 'needs_relevance_and_source_review'}
        audit.append({**row, 'exclusion_reason': reason})
        if not reason:
            rows.append(row)
    return rows, audit


def select(rows, limit):
    # Exact URL/title dedup within each scope. This is not event-level merging.
    ordered = sorted(rows, key=lambda r: (r['published_at_utc'], r['article_id']), reverse=True)
    chosen, urls, titles = [], set(), set()
    for row in ordered:
        key = re.sub(r'\W+', ' ', row['title'].lower()).strip()
        if row['google_news_url'] in urls or key in titles:
            continue
        urls.add(row['google_news_url'])
        titles.add(key)
        chosen.append(row)
        if len(chosen) >= limit:
            break
    return chosen if limit else []


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--days', type=int, default=7)
    p.add_argument('--national-limit', type=int, default=5)
    p.add_argument('--state-limit', type=int, default=3)
    p.add_argument('--states', default=','.join(STATES), help='Comma-separated state codes')
    p.add_argument('--until', help='Timezone-aware ISO cutoff; default now (UTC)')
    p.add_argument('--out', type=Path, help='Output root; default project data/news_google')
    args = p.parse_args()
    codes = list(dict.fromkeys(x.strip().upper() for x in args.states.split(',') if x.strip()))
    if not codes or set(codes) - STATES.keys():
        p.error('Use supported Senate state codes, for example MI,PA.')
    if args.days < 1 or min(args.national_limit, args.state_limit) < 0:
        p.error('days must be positive and limits must be nonnegative.')
    now = datetime.now(timezone.utc)
    cutoff = now
    if args.until:
        try:
            cutoff = datetime.fromisoformat(args.until.replace('Z', '+00:00'))
            if cutoff.tzinfo is None:
                raise ValueError('Missing timezone')
            cutoff = cutoff.astimezone(timezone.utc)
        except ValueError:
            p.error('--until requires an ISO datetime with timezone.')
    if cutoff > now:
        p.error('--until cannot be in the future.')
    start = cutoff - timedelta(days=args.days)
    # Supports placement in the repo root or its scripts directory.
    parent = Path(__file__).resolve().parent
    project = parent.parent if parent.name == 'scripts' else parent
    root = args.out or project / 'data' / 'news_google'
    root.mkdir(parents=True, exist_ok=True)
    selected, coverage = [], []
    scopes = [('national', '', args.national_limit)] + [('state', STATES[c], args.state_limit) for c in codes]
    for number, (level, state, limit) in enumerate(scopes):
        if not limit:
            continue
        if number:
            time.sleep(2)
        label = state or 'United States'
        # Broader calendar-date query, then exact timestamp filtering locally.
        after = (start - timedelta(days=1)).strftime('%Y-%m-%d')
        before = (cutoff + timedelta(days=1)).strftime('%Y-%m-%d')
        q = f'"{label}" {ISSUES} after:{after} before:{before}'
        url = 'https://news.google.com/rss/search?' + urlencode(
            {'q': q, 'hl': 'en-US', 'gl': 'US', 'ceid': 'US:en'})
        status = {'level': level, 'state': state, 'status': 'ok',
                  'returned': 0, 'eligible': 0, 'selected': 0, 'error': ''}
        print(f'Collecting {label} (up to {limit})...', flush=True)
        try:
            raw = fetch(url)
            rows, audit = parse_feed(raw, level, state, start, cutoff)
            chosen = select(rows, limit)
            selected.extend(chosen)
            status.update(returned=len(audit), eligible=len(rows), selected=len(chosen))
            if not audit:
                status['status'] = 'empty_feed'
            elif not chosen:
                status['status'] = 'no_eligible_items'
        except Exception as exc:
            status.update(status='collection_failed', error=str(exc))
            print(f'  Failed: {exc}', flush=True)
        coverage.append(status)
        write_csv(root / 'news_google.csv', selected, FIELDS)
        write_csv(root / 'coverage.csv', coverage,
                  ['level', 'state', 'status', 'returned', 'eligible', 'selected', 'error'])
    print(f'Saved {len(selected)} selected records to {root.resolve()}')
    print('RSS metadata only. Review original articles before using as voter context.')
    return int(any(r['status'] == 'collection_failed' for r in coverage))


if __name__ == '__main__':
    sys.exit(main())
