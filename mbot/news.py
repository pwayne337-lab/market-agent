"""Counts of returned Yahoo articles in a fixed window, not all published news."""
from datetime import timedelta
import time
from typing import Dict, List
import pandas as pd

METHOD = 'yahoo_returned_items_v2'
FETCH_LIMIT = 100

def fetch_headlines(symbols: List[str], *, end, pause=0.05, detailed=False):
    import yfinance as yf
    end = pd.Timestamp(end)
    since = end - timedelta(days=1)
    counts, titles, failed, capped = {}, {}, [], []
    for sym in symbols:
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
            # A capped sample still has useful dated headlines. Only its count
            # is unknown: it must never enter the burst baseline as a total.
            if detailed:
                titles[sym] = ts[:20]
            if len(items) >= FETCH_LIMIT and (not dates or min(dates) > since):
                capped.append(sym)
                continue  # The returned sample does not reach the window boundary.
            counts[sym], titles[sym] = len(ts), ts
        except Exception:
            failed.append(sym)
        finally:
            time.sleep(pause)
    return counts, titles, failed, capped
