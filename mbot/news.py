"""Counts of returned Yahoo articles in a fixed window, not all published news."""
from datetime import timedelta
import time
from typing import Dict, List
import pandas as pd

METHOD = 'yahoo_returned_items_v2'
FETCH_LIMIT = 100


def _error_reason(exc):
    """Keep useful failure categories without persisting provider URLs or secrets."""
    known = {'invalid news response', 'invalid article',
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
    Diagnostic collection does not alter counts, baselines or trading rules.
    """
    import yfinance as yf
    end = pd.Timestamp(end)
    since = end - timedelta(days=1)
    counts, titles, failed, capped = {}, {}, [], []
    observations = {}
    for sym in symbols:
        items, ts, dates = None, [], []
        error_type, error_reason = None, None
        parsed = False
        try:
            items = yf.Ticker(sym).get_news(count=FETCH_LIMIT)
            if not isinstance(items, list):
                raise ValueError('invalid news response')
            seen, ts, dates = set(), [], []
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError('invalid article')
                content = item.get('content') or item
                title = content.get('title') or ''
                when = content.get('pubDate') or item.get('providerPublishTime')
                if not title or not when:
                    raise ValueError('article missing date or title')
                dt = (pd.Timestamp(when, unit='s', tz='UTC') if isinstance(when, (int, float))
                      else pd.Timestamp(when))
                if dt.tzinfo is None:
                    raise ValueError('article timestamp has no timezone')
                dates.append(dt)
                key = content.get('id') or item.get('id') or (title, dt.isoformat())
                if since < dt <= end and key not in seen:
                    seen.add(key)
                    provider = content.get('provider') or {}
                    link = content.get('canonicalUrl') or content.get('clickThroughUrl') or {}
                    url = link.get('url') if isinstance(link, dict) else None
                    url = url or content.get('link') or item.get('link')
                    if not isinstance(url, str) or not url.startswith(('https://', 'http://')):
                        url = None
                    ts.append({'title': str(title).strip()[:500], 'published_at': dt.isoformat(),
                               'publisher': str(provider.get('displayName') or content.get('publisher') or '')[:200],
                               'url': url, 'source': 'Yahoo Finance'} if detailed else title)
            parsed = True
            # A capped sample still has useful dated headlines. Only its count
            # is unknown: it must never enter the burst baseline as a total.
            if detailed:
                titles[sym] = ts[:20]
            if len(items) >= FETCH_LIMIT and (not dates or min(dates) > since):
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
                    'sample_reaches_window_start': bool(dates and min(dates) <= since),
                    'observed_zero_in_window': parsed and not ts,
                    'published_news_coverage_confirmed': False,
                    'status': ('failed' if error_type else 'capped' if sym in capped else
                               'empty_response' if not items else
                               'observed_in_window' if ts else 'no_items_in_window'),
                    'error_type': error_type, 'error_reason': error_reason,
                }
            time.sleep(pause)
    if diagnostics is not None:
        matched = sum(row['matched_in_window_count'] for row in observations.values())
        universe_zero = len(observations) > 1 and matched == 0
        diagnostics.clear()
        diagnostics.update({
            'schema_version': 1, 'window_start': since.isoformat(),
            'window_end': end.isoformat(), 'requested_symbol_count': len(observations),
            'observed_matched_item_count': matched,
            'universe_zero_in_window_observed': universe_zero,
            'published_news_coverage_confirmed': False,
            'symbols': observations,
            'warnings': (['No in-window headlines observed across the requested universe; '
                          'news coverage is unconfirmed, not evidence of no published news.']
                         if universe_zero else []),
        })
    return counts, titles, failed, capped
