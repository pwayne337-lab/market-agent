"""
The event record: what happened, and what the price did next.

"Learn how the market moves" cannot be done by watching. It is done by
writing down, on the day, that something happened -- a move too large for
that stock, a burst of headlines, or both -- and then coming back one and
five sessions later to write down what the price did after. Once there are
enough of those, questions get answers with sample sizes attached:

    After a 3-ATR up day on heavy news, what did the next five sessions do?
    After a 3-ATR down day with NO news, did it keep falling or bounce?

Those answers are the agent's understanding of how the market moves. They
are facts about this universe over this period, nothing more, and every one
is printed with the count it came from. Under `min_sample` the record shows
the count and refuses to draw a conclusion.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import MarketConfig
from .sessions import follow_dates, last_completed

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
EVENTS_FILE = STATE_DIR / "events.jsonl"


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def detect(bars: Dict[str, pd.DataFrame], news_counts: Dict[str, int],
           news_baseline: Dict[str, float], headlines: Dict[str, List[str]],
           cfg: MarketConfig, as_of: Optional[str] = None) -> List[dict]:
    """Today's events across the universe, from the last completed bar.

    An event is a move of at least `event_move_atr` ATRs (up or down), a
    headline count of at least `news_spike_multiple` times the symbol's
    30-day average, or both. Each is written with the numbers that made it
    one, so the record can later be cut by kind.
    """
    out = []
    for sym, df in bars.items():
        if df is None or len(df) < cfg.atr_period + 2:
            continue
        c = df["close"].astype(float)
        atr = _atr(df, cfg.atr_period)
        a = float(atr.iloc[-1])
        if not a or a != a or a <= 0:
            continue
        day = df.index[-1]
        if as_of and str(day.date()) != as_of:
            continue   # stale bar: not today's event
        move = float(c.iloc[-1] - c.iloc[-2])
        move_atr = move / a
        move_pct = (float(c.iloc[-1]) / float(c.iloc[-2]) - 1) * 100
        gap_pct = (float(df["open"].iloc[-1]) / float(c.iloc[-2]) - 1) * 100
        n_news = int(news_counts.get(sym, 0))
        base = float(news_baseline.get(sym, 0.0))
        news_spike = base > 0 and n_news >= cfg.news_spike_multiple * base
        big_move = abs(move_atr) >= cfg.event_move_atr
        if not (big_move or news_spike):
            continue
        kind = ("move+news" if big_move and news_spike
                else "move" if big_move else "news")
        out.append({
            "date": str(day.date()), "symbol": sym, "kind": kind, "rules_version": 2,
            "direction": "up" if move > 0 else "down",
            "move_pct": round(move_pct, 2), "move_atr": round(move_atr, 2),
            "gap_pct": round(gap_pct, 2), "close": round(float(c.iloc[-1]), 2),
            "headlines_today": n_news, "headlines_usual": round(base, 2),
            "news_assessment": ("assessed" if sym in news_counts and sym in news_baseline else
                                "unavailable" if sym not in news_counts else "warming up"),
            "top_headlines": (headlines.get(sym) or [])[:3],
            "follow": {},
        })
    out.sort(key=lambda e: -abs(e["move_atr"]))
    return out


def load_events() -> List[dict]:
    if not EVENTS_FILE.exists():
        return []
    rows = []
    for line in EVENTS_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def save_events(rows: List[dict]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    EVENTS_FILE.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))


def merge(existing: List[dict], fresh: List[dict]) -> List[dict]:
    """Update repeated finalized reads without duplicating date/symbol keys."""
    by_key = {(e.get("date"), e.get("symbol")): e for e in existing}
    for e in fresh:
        key = (e.get("date"), e.get("symbol"))
        old = by_key.get(key, {})
        by_key[key] = dict(e, follow=old.get("follow") or {})
    return sorted(by_key.values(), key=lambda e: (e.get("date", ""), e.get("symbol", "")))


def fill_follow_through(events: List[dict], bars: Dict[str, pd.DataFrame],
                        cfg: MarketConfig) -> int:
    """Write in what happened N sessions after each event, once N sessions
    have completed; reruns repair corrected source bars. Returns how many fields were filled this run."""
    filled = 0
    completed = last_completed()
    for e in events:
        df = bars.get(e.get("symbol"))
        if df is None or pd.Timestamp(e["date"]) not in df.index:
            continue
        c = df["close"].astype(float)
        base = float(c.loc[pd.Timestamp(e["date"])])
        for n in cfg.follow_through_days:
            key = f"{n}d"
            target = follow_dates(e["date"], n)
            if target > completed or pd.Timestamp(target) not in c.index:
                continue
            value = float(c.loc[pd.Timestamp(target)])
            if not (math.isfinite(value) and math.isfinite(base) and base > 0):
                continue
            # Recompute from final bars so reruns can repair source corrections.
            result = round((value / base - 1) * 100, 2)
            if (e.get("follow") or {}).get(key) != result:
                e.setdefault("follow", {})[key] = result
                filled += 1
    return filled


def _stats(vals: List[float], n_min: int) -> dict:
    vals = [v for v in vals if v is not None and v == v]
    n = len(vals)
    if n == 0:
        return {"n": 0}
    out = {"n": n}
    if n < n_min:
        out["note"] = f"not enough data ({n} of {n_min})"
        return out
    arr = np.array(vals, dtype=float)
    mean = float(arr.mean())
    out.update({"mean_pct": round(mean, 2), "median_pct": round(float(np.median(arr)), 2),
                "up_share_pct": round(float((arr > 0).mean() * 100), 1),
                "descriptive_only": True})
    return out


def what_the_record_says(events: List[dict], cfg: MarketConfig) -> List[dict]:
    """Follow-through by kind and direction, with sample sizes and descriptive results. This is the
    agent's understanding of how the market moves, and it is only as good
    as the count next to each line."""
    groups: Dict[str, List[dict]] = {}
    for e in events:
        key = (f"{e.get('kind')} / {e.get('direction')} / rules {e.get('rules_version', 1)}"
               f" / news {e.get('news_assessment', 'legacy')}")
        groups.setdefault(key, []).append(e)
    out = []
    for key, evs in sorted(groups.items()):
        row = {"group": key, "events": len(evs),
               "distinct_dates": len({e.get("date") for e in evs}),
               "distinct_symbols": len({e.get("symbol") for e in evs})}
        for n in cfg.follow_through_days:
            vals = [(e.get("follow") or {}).get(f"{n}d") for e in evs]
            row[f"after_{n}d"] = _stats(vals, cfg.min_sample)
            row[f"after_{n}d"]["distinct_dates"] = len({e.get("date") for e in evs if (e.get("follow") or {}).get(f"{n}d") is not None})
        out.append(row)
    return out
