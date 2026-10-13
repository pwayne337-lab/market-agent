"""Counts of returned Yahoo articles in a fixed window, not all published news."""
from datetime import timedelta
import time
from typing import Dict, List
from urllib.parse import urlparse
import pandas as pd

METHOD = 'yahoo_search_returned_items_v1'
FETCH_LIMIT = 50
VALIDATION_VERSION = 1
REQUEST_TIMEOUT = 10
FETCH_BUDGET = 180
SEARCH_URL = 'https://query1.finance.yahoo.com/v1/finance/search'


def _search_news(symbol, observation):
    """Read Yahoo's working search feed; its response explicitly tags symbols.

    curl_cffi is already supplied by the pinned yfinance dependency. There is
    one bounded request, no provider retry or paid service, and no credentials.
    """
    from curl_cffi import requests
    response = requests.get(SEARCH_URL, params={
        'q': symbol, 'quotesCount': 0, 'newsCount': FETCH_LIMIT,
        'enableFuzzyQuery': 'false',
    }, impersonate='chrome', timeout=REQUEST_TIMEOUT)
    status = response.status_code
    observation.update(http_status=status, http_success=200 <= status < 300,
                       backend='yahoo_search', sample_limit=FETCH_LIMIT)
    if not observation['http_success']:
        raise ValueError('provider HTTP failure')
    payload = response.json()
    valid = (isinstance(payload, dict) and isinstance(payload.get('news'), list)
             and not payload.get('error') and not payload.get('errors'))
    observation['payload_valid'] = valid
    if not valid:
        raise ValueError('invalid provider news payload')
    return payload['news']


def _error_reason(exc):
    """Keep useful failure categories without persisting provider URLs or secrets."""
    known = {'invalid news response', 'provider HTTP failure', 'invalid provider news payload', 'invalid article',
             'article missing symbol relevance',
             'article missing publisher',
             'headline batch budget exhausted',
             'article missing date or title', 'article timestamp has no timezone'}
    if isinstance(exc, ValueError) and str(exc) in known:
        return str(exc)
    name = type(exc).__name__.lower()
    if 'timeout' in name:
        return 'Provider request timed out; raw details omitted.'
    if 'connection' in name:
        return 'Provider connection failed; raw details omitted.'
    if 'http' in name:
        return 'Provider HTTP request failed; raw details omitted.'
    return 'Provider response could not be processed; raw details omitted.'


def fetch_headlines(symbols: List[str], *, end, pause=0.05, detailed=False,
                    diagnostics=None):
    """Return the existing four values; optionally fill a diagnostic dictionary.

    A zero is an observed count in Yahoo's returned sample. It never confirms
    that no news was published, even when the sample reaches the window start.
    Observed HTTP failures and malformed provider payloads are unavailable.
    Search results must explicitly name the requested symbol as related.
    """
    end = pd.Timestamp(end)
    since = end - timedelta(days=1)
    counts, titles, failed, capped = {}, {}, [], []
    observations = {}
    deadline = time.monotonic() + FETCH_BUDGET
    for sym in symbols:
        items, ts, dates, related_dates = None, [], [], []
        provider_response = {}
        error_type, error_reason = None, None
        parsed = False
        try:
            if time.monotonic() >= deadline:
                provider_response['request_skipped'] = 'batch_budget'
                raise ValueError('headline batch budget exhausted')
            items = _search_news(sym, provider_response)
            if not isinstance(items, list):
                raise ValueError('invalid news response')
            seen, ts, dates = set(), [], []
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError('invalid article')
                content = item.get('content') or item
                related = item.get('relatedTickers')
                if not isinstance(related, list) or any(not isinstance(s, str) for s in related):
                    raise ValueError('article missing symbol relevance')
                title = content.get('title') or ''
                when = content.get('pubDate') or item.get('providerPublishTime')
                if not isinstance(title, str) or not title.strip() or not when:
                    raise ValueError('article missing date or title')
                provider = content.get('provider') or {}
                publisher = provider.get('displayName') or content.get('publisher')
                if not isinstance(publisher, str) or not publisher.strip():
                    raise ValueError('article missing publisher')
                dt = (pd.Timestamp(when, unit='s', tz='UTC') if isinstance(when, (int, float))
                      else pd.Timestamp(when))
                if dt.tzinfo is None:
                    raise ValueError('article timestamp has no timezone')
                dates.append(dt)
                if sym in related:
                    related_dates.append(dt)
                key = content.get('id') or item.get('uuid') or item.get('id') or (title, dt.isoformat())
                if sym in related and since < dt <= end and key not in seen:
                    seen.add(key)
                    link = content.get('canonicalUrl') or content.get('clickThroughUrl') or {}
                    url = link.get('url') if isinstance(link, dict) else None
                    url = url or content.get('link') or item.get('link')
                    if (not isinstance(url, str) or urlparse(url).scheme not in ('https', 'http')
                            or not urlparse(url).netloc):
                        url = None
                    ts.append({'title': str(title).strip()[:500], 'published_at': dt.isoformat(),
                               'publisher': publisher[:200],
                               'url': url, 'source': 'Yahoo Finance', 'related_symbols': related} if detailed else title)
            parsed = True
            # A capped sample still has useful dated headlines. Only its count
            # is unknown: it must never enter the burst baseline as a total.
            if detailed:
                titles[sym] = ts[:20]
            if len(items) >= FETCH_LIMIT and (not related_dates or min(related_dates) > since
                                             or dates != sorted(dates, reverse=True)):
                capped.append(sym)
                continue  # The returned sample does not reach the window boundary.
            counts[sym], titles[sym] = len(ts), ts
        except Exception as exc:
            failed.append(sym)
            error_type, error_reason = type(exc).__name__, _error_reason(exc)
        finally:
            if diagnostics is not None:
                observations[sym] = {
                    'returned_item_count': len(items) if isinstance(items, list) else None,
                    'parsed_item_count': len(dates), 'parse_complete': parsed,
                    'matched_in_window_count': len(ts),
                    'oldest_returned_at': min(dates).isoformat() if dates else None,
                    'newest_returned_at': max(dates).isoformat() if dates else None,
                    'related_item_count': len(related_dates),
                    'returned_newest_first': dates == sorted(dates, reverse=True),
                    'sample_reaches_window_start': bool(related_dates and min(related_dates) <= since),
                    'observed_zero_in_window': parsed and not ts,
                    'published_news_coverage_confirmed': False,
                    'status': ('failed' if error_type else 'capped' if sym in capped else
                               'empty_response' if not items else
                               'observed_in_window' if ts else 'no_items_in_window'),
                    'error_type': error_type, 'error_reason': error_reason,
                    'provider_response': provider_response or None,
                }
            if time.monotonic() < deadline:
                time.sleep(pause)
    if diagnostics is not None:
        matched = sum(row['matched_in_window_count'] for row in observations.values())
        universe_zero = len(observations) > 1 and matched == 0
        http_errors = sum((row.get('provider_response') or {}).get('http_success') is False
                          for row in observations.values())
        notes = (['No in-window headlines observed across the requested universe; '
                  'news coverage is unconfirmed, not evidence of no published news.']
                 if universe_zero else [])
        if http_errors:
            notes.append(f'News provider returned unsuccessful HTTP status for {http_errors} symbols; '
                         'affected samples are unavailable and excluded from new baseline observations.')
        diagnostics.clear()
        diagnostics.update({
            'schema_version': 1, 'window_start': since.isoformat(),
            'window_end': end.isoformat(), 'requested_symbol_count': len(observations),
            'observed_matched_item_count': matched,
            'universe_zero_in_window_observed': universe_zero,
            'published_news_coverage_confirmed': False,
            'provider_http_error_symbol_count': http_errors,
            'symbols': observations,
            'warnings': notes,
        })
    return counts, titles, failed, capped
