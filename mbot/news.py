"""
Headline counts, from the same free source the trading agent reads.

No model reads these. What the agent keeps is how MANY headlines a name
drew today against its usual, because a burst of coverage is a measurable
thing and "the news was bad" is not. The titles are kept so a person can
read what the burst was about.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple


def fetch_headlines(symbols: List[str], days: int = 1, pause: float = 0.05
                    ) -> Tuple[Dict[str, int], Dict[str, List[str]], List[str]]:
    """Headline count per symbol over the last `days`, plus titles, plus the
    symbols the source failed on (so silence is not mistaken for quiet)."""
    try:
        import yfinance as yf
    except ImportError:
        return {}, {}, list(symbols)
    since = datetime.now(timezone.utc) - timedelta(days=days)
    counts: Dict[str, int] = {}
    titles: Dict[str, List[str]] = {}
    failed: List[str] = []
    for sym in symbols:
        try:
            items = yf.Ticker(sym).news or []
        except Exception:
            failed.append(sym)
            continue
        n, ts = 0, []
        for item in items:
            content = item.get("content") if isinstance(item, dict) else None
            title = (content or {}).get("title") or item.get("title") or ""
            when = (content or {}).get("pubDate") or item.get("providerPublishTime")
            try:
                if isinstance(when, (int, float)):
                    dt = datetime.fromtimestamp(when, tz=timezone.utc)
                else:
                    dt = datetime.fromisoformat(str(when).replace("Z", "+00:00"))
            except Exception:
                continue
            if dt >= since and title:
                n += 1
                ts.append(title)
        counts[sym] = n
        titles[sym] = ts
        time.sleep(pause)
    return counts, titles, failed
