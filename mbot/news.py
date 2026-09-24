"""Counts of returned Yahoo articles in a fixed window, not all published news."""
from datetime import timedelta
import time
from typing import Dict, List
import pandas as pd

METHOD = 'yahoo_returned_items_v2'
FETCH_LIMIT = 100

def fetch_headlines(symbols: List[str], *, end, pause=0.05):
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
                    ts.append(title)
            if len(items) >= FETCH_LIMIT and (not dates or min(dates) > since):
                capped.append(sym)
                continue  # The returned sample does not reach the window boundary.
            counts[sym], titles[sym] = len(ts), ts
        except Exception:
            failed.append(sym)
        finally:
            time.sleep(pause)
    return counts, titles, failed, capped
