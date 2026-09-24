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
from mbot import report, sessions
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
    out = {}
    for symbol in symbols:
        vals = [float(r["counts"][symbol]) for r in history[-days:]
                if r.get("method") == newsmod.METHOD and symbol in r.get("counts", {})]
        if len(vals) >= 5:
            out[symbol] = sum(vals) / len(vals)
    return out


def _previous():
    try:
        return json.loads(READ_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _publish_failure(errors, status="unavailable"):
    prev = _previous()
    last_good = prev if prev.get("healthy") and prev.get("finalized") else prev.get("last_good")
    if last_good:
        last_good = {k: v for k, v in last_good.items() if k != "last_good"}
    read = {"schema_version": 2, "updated_at": now_iso(), "healthy": False,
            "finalized": False, "status": status, "errors": errors,
            "risk": {"word": "unknown", "score": 0, "of": 4, "checks": []},
            "last_good": last_good}
    _save(read)
    report.write_site(read)
    for error in errors:
        print(f"WARNING: {error}")


def cmd_run(args):
    try:
        return _run(args)
    except Exception as exc:
        _publish_failure([f"Market read failed: {type(exc).__name__}: {exc}"])
        return 1


def _run(args):
    cfg = MarketConfig()
    session = sessions.today_session()
    if session is None or not session["ready"]:
        # Holidays and early manual runs never change the permanent research record.
        prev = _previous()
        if prev.get("finalized") and prev.get("as_of") == sessions.last_completed():
            print("No new completed session; retaining the dated final read.")
            return 0
        _publish_failure(["Waiting for a completed NYSE session and a 15-minute settlement buffer."], "waiting")
        return 0
    as_of = session["date"]
    prev = _previous()
    if (getattr(args, "scheduled", False) and prev.get("healthy") and prev.get("finalized")
            and prev.get("as_of") == as_of and prev.get("schema_version") == 2):
        print(f"Completed healthy read already recorded for {as_of}; retry skipped.")
        return 0
    universe = sorted(set(cfg.watchlist) | set(INDEXES) | set(SECTORS) | set(RATES_AND_FEAR))
    print(f"Loading {len(universe)} symbols for completed session {as_of}...")
    if args.cached:
        available = [sym for sym in universe if datamod._cache_path(sym).exists()]
        if not available:
            raise ValueError("no cached prices")
        bars = datamod.load_universe(available, start=cfg.history_start, refresh=False)
    else:
        bars = datamod.load_universe(universe, start=cfg.history_start, refresh=True)
    bars = {sym: df.loc[df.index <= as_of] for sym, df in bars.items()}
    stale = [sym for sym, df in bars.items() if df.empty or str(df.index[-1].date()) != as_of]
    bars = {sym: df for sym, df in bars.items() if sym not in stale}
    missing = sorted(set(universe) - set(bars))
    errors = []
    if missing:
        errors.append(f"Missing or stale prices for {len(missing)} symbols: {', '.join(missing[:10])}")
    spy = bars.get(cfg.regime_symbol)
    if spy is None or len(spy) < cfg.sma_slow + 1:
        raise ValueError(f"{cfg.regime_symbol} has no complete, current history for {as_of}")
    idx = regime.index_read(spy, cfg)
    indexes = {s: regime.index_read(bars[s], cfg) for s in INDEXES if s in bars}
    br = regime.breadth(bars, cfg.watchlist, cfg)
    br["coverage_pct"] = round(br.get("counted", 0) / len(cfg.watchlist) * 100, 1)
    if br["coverage_pct"] < 90:
        errors.append("Breadth coverage below 90%; breadth check is unknown")
    vol = regime.volatility(spy, bars.get("^VIX"), cfg)
    if "vix" not in vol:
        errors.append("VIX unavailable; volatility check is unknown")
    # Relative returns must cover identical sessions, not just equal row counts.
    aligned = {s: df.reindex(spy.index).dropna() for s, df in bars.items()}
    for sym in list(aligned):
        if not aligned[sym].index[-(cfg.rs_long + 1):].equals(spy.index[-(cfg.rs_long + 1):]):
            aligned.pop(sym)
    secs = regime.sector_strength(aligned, cfg)
    if len(secs) != len(SECTORS):
        errors.append("Some sector comparisons have incomplete session coverage")
    risk = regime.risk_score(idx, br if br["coverage_pct"] >= 90 else {}, vol, cfg)

    counts, titles, failed, capped = {}, {}, [], []
    history = _load_news_history()
    baseline = _news_baseline([h for h in history if h.get("date", "") < as_of],
                              cfg.watchlist, cfg.news_baseline_days)
    if not (args.no_news or args.cached):
        counts, titles, failed, capped = newsmod.fetch_headlines(cfg.watchlist, end=session["close"])
        if failed:
            errors.append(f"Headlines unavailable for {len(failed)} symbols")
        # Capped responses are omitted, not interpreted as a quiet day or a known total.
        if capped:
            errors.append(f"Headline sample limit reached for {len(capped)} symbols; excluded from baseline")
        if counts:
            old = next((h for h in history if h.get("date") == as_of and h.get("method") == newsmod.METHOD), {})
            row = {"date": as_of, "method": newsmod.METHOD, "window_end": session["close"].isoformat(),
                   "counts": dict(old.get("counts", {}), **counts)}
            history = sorted([h for h in history if h.get("date") != as_of] + [row], key=lambda h: h["date"])
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            NEWS_FILE.write_text("".join(json.dumps(h) + "\n" for h in history))
    else:
        errors.append("Headlines skipped; news events were not assessed")
    watch_bars = {s: bars[s] for s in cfg.watchlist if s in bars}
    today = ev.detect(watch_bars, counts, baseline, titles, cfg, as_of=as_of)
    # Only finalized v2 events contribute to the new research series.
    record = ev.merge(ev.load_events(), today)
    filled = ev.fill_follow_through(record, watch_bars, cfg)
    ev.save_events(record)
    says = ev.what_the_record_says(record, cfg)
    read = {
        "schema_version": 2, "rules_version": 2, "updated_at": now_iso(), "as_of": as_of,
        "session_close": session["close"].isoformat(), "expires_at": sessions.next_expiry(as_of),
        "finalized": True, "healthy": not errors, "status": "complete" if not errors else "degraded", "errors": errors,
        "risk": risk, "index": idx, "indexes": indexes, "breadth": br,
        "volatility": vol, "sectors": secs, "events_today": today,
        "record": {"events_total": len(record), "with_5d": sum("5d" in (e.get("follow") or {}) for e in record),
                   "filled_this_run": filled, "what_it_says": says},
        "news": {"fetched": bool(counts), "baseline_days": len([h for h in history if h.get("method") == newsmod.METHOD]),
                 "baseline_ready": len(baseline) == len(cfg.watchlist), "ready_symbols": len(baseline),
                 "valid_symbols": len(counts), "failed_symbols": failed, "capped_symbols": capped,
                 "method": newsmod.METHOD, "note": "Returned Yahoo articles in the 24 hours ending at the close; not all published news."},
        "min_sample": cfg.min_sample, "universe": {"watchlist": len(cfg.watchlist), "loaded": len(bars)},
    }
    if errors:
        read["last_good"] = (prev if prev.get("healthy") and prev.get("finalized") else prev.get("last_good"))
    read["brief"] = report.brief(read)
    _save(read)
    print("\n" + read["brief"] + "\n")
    print(f"Page: {report.write_site(read)}")
    return 0 if read["healthy"] else 1


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
    r.add_argument("--scheduled", action="store_true", help="skip an already successful session")
    r.set_defaults(func=cmd_run)
    p = sub.add_parser("page", help="rebuild the page from the saved read")
    p.set_defaults(func=cmd_page)
    c = sub.add_parser("record", help="what the event record says so far")
    c.set_defaults(func=cmd_record)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
