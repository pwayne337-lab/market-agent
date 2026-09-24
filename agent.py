#!/usr/bin/env python3
"""
The market agent. It trades nothing. Each evening it reads the same names the
trading agent can buy, plus the indexes and sectors around them, and writes
down what the market did and what happened to prices after things happened.

    python agent.py run            read the market, record events, publish
    python agent.py run --no-news  same, without fetching headlines
    python agent.py page           rebuild the page from the saved read
    python agent.py record         print what the event record says so far
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from mbot import data as datamod
from mbot import events as ev
from mbot import news as newsmod
from mbot import regime
from mbot import report
from mbot.config import INDEXES, RATES_AND_FEAR, SECTORS, MarketConfig

STATE_DIR = Path(__file__).resolve().parent / "state"
READ_FILE = STATE_DIR / "market.json"
NEWS_FILE = STATE_DIR / "news_counts.jsonl"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_news_history():
    if not NEWS_FILE.exists():
        return []
    rows = []
    for line in NEWS_FILE.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _news_baseline(history, symbols, days):
    """Average daily headline count per symbol over the last `days` recorded
    days. Yahoo only serves recent headlines, so the agent builds its own
    history one day at a time; until five days exist, no baseline, and no
    news-spike events are called."""
    recent = history[-days:]
    if len(recent) < 5:
        return {}
    out = {}
    for s in symbols:
        vals = [float(r.get("counts", {}).get(s, 0)) for r in recent]
        out[s] = sum(vals) / len(vals)
    return out


def cmd_run(args):
    cfg = MarketConfig()
    universe = sorted(set(cfg.watchlist) | set(INDEXES) | set(SECTORS) | set(RATES_AND_FEAR))
    if args.cached:
        # Offline: only what is already on disk, and no attempt to fetch the
        # rest. Used by the tests and by anyone without Yahoo access.
        universe = [s for s in universe if datamod._cache_path(s).exists()]
    print(f"Loading {len(universe)} symbols from {cfg.history_start}...")
    bars = datamod.load_universe(universe, start=cfg.history_start, refresh=not args.cached)
    missing = sorted(set(universe) - set(bars))
    errors = []
    if missing:
        errors.append(f"no data for {len(missing)} symbol(s): {', '.join(missing[:8])}"
                      + (" ..." if len(missing) > 8 else ""))
    spy = bars.get(cfg.regime_symbol)
    if spy is None:
        print(f"Cannot read {cfg.regime_symbol}; nothing to say about the market.")
        errors.append(f"{cfg.regime_symbol} unavailable")
        st = {"updated_at": now_iso(), "healthy": False, "errors": errors}
        _save(st)
        return
    as_of = str(spy.index[-1].date())

    idx = regime.index_read(spy, cfg)
    indexes = {s: regime.index_read(bars[s], cfg) for s in INDEXES if s in bars}
    br = regime.breadth(bars, cfg.watchlist, cfg)
    vol = regime.volatility(spy, bars.get("^VIX"), cfg)
    secs = regime.sector_strength(bars, cfg)
    risk = regime.risk_score(idx, br, vol, cfg)

    counts, titles, failed = {}, {}, []
    history = _load_news_history()
    if not args.no_news:
        counts, titles, failed = newsmod.fetch_headlines(cfg.watchlist)
        if failed:
            errors.append(f"headlines unavailable for {len(failed)} symbol(s)")
        if counts:
            history.append({"date": as_of, "counts": counts})
            history = [h for h in history if h.get("date")]
            # one row per date; the latest read for a date wins
            dedup = {}
            for h in history:
                dedup[h["date"]] = h
            history = [dedup[k] for k in sorted(dedup)]
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            NEWS_FILE.write_text("".join(json.dumps(h) + "\n" for h in history))
    baseline = _news_baseline(history[:-1] if counts else history, cfg.watchlist,
                              cfg.news_baseline_days)

    watch_bars = {s: bars[s] for s in cfg.watchlist if s in bars}
    today = ev.detect(watch_bars, counts, baseline, titles, cfg, as_of=as_of)
    record = ev.merge(ev.load_events(), today)
    filled = ev.fill_follow_through(record, watch_bars, cfg)
    ev.save_events(record)
    says = ev.what_the_record_says(record, cfg)

    read = {
        "updated_at": now_iso(), "as_of": as_of, "healthy": not errors, "errors": errors,
        "risk": risk, "index": idx, "indexes": indexes, "breadth": br,
        "volatility": vol, "sectors": secs, "events_today": today,
        "record": {"events_total": len(record),
                   "with_5d": sum(1 for e in record if "5d" in (e.get("follow") or {})),
                   "filled_this_run": filled, "what_it_says": says},
        "news": {"fetched": bool(counts), "baseline_days": len(history),
                 "baseline_ready": bool(baseline)},
        "min_sample": cfg.min_sample,
        "universe": {"watchlist": len(cfg.watchlist), "loaded": len(bars)},
    }
    read["brief"] = report.brief(read)
    _save(read)
    print("\n" + read["brief"] + "\n")
    for e in errors:
        print(f"  WARNING: {e}")
    print(f"Page: {report.write_site(read)}")


def _save(read):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    READ_FILE.write_text(json.dumps(read, indent=2, default=str))


def cmd_page(args):
    if not READ_FILE.exists():
        print("No read saved yet. Run `python agent.py run` first.")
        return
    read = json.loads(READ_FILE.read_text())
    print(f"Page: {report.write_site(read)}")


def cmd_record(args):
    cfg = MarketConfig()
    record = ev.load_events()
    print(f"{len(record)} event(s) on record.")
    for row in ev.what_the_record_says(record, cfg):
        print(f"  {row['group']:22} n={row['events']:4}  "
              + "  ".join(f"{k}: {v.get('note') or (str(v.get('mean_pct')) + '% avg')}"
                          for k, v in row.items() if k.startswith("after_")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="read the market, record today's events, publish the page")
    r.add_argument("--no-news", action="store_true", help="skip headline fetching")
    r.add_argument("--cached", action="store_true", help="use cached prices only (offline)")
    r.set_defaults(func=cmd_run)
    p = sub.add_parser("page", help="rebuild the page from the saved read")
    p.set_defaults(func=cmd_page)
    c = sub.add_parser("record", help="what the event record says so far")
    c.set_defaults(func=cmd_record)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
