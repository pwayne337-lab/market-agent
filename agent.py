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
import pandas as pd

from mbot import data as datamod
from mbot import events as ev
from mbot import news as newsmod
from mbot import regime
from mbot import report, sessions, findings
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
    current = [r for r in history if r.get("method") == newsmod.METHOD]
    for symbol in symbols:
        vals = [float(r["counts"][symbol]) for r in current[-days:]
                if symbol in r.get("counts", {})]
        if len(vals) >= 5:
            out[symbol] = sum(vals) / len(vals)
    return out


def _previous():
    try:
        read = json.loads(READ_FILE.read_text())
        return read if isinstance(read, dict) else {}
    except (OSError, ValueError):
        return {}


def _complete_news(read, symbols):
    coverage = (read.get("stock_research") or {}).get("coverage") or {}
    diagnostics = (read.get("news") or {}).get("diagnostics") or {}
    return bool(
        type((read.get("news") or {}).get("validation_version")) is int
        and (read.get("news") or {}).get("validation_version") == newsmod.VALIDATION_VERSION
        and (read.get("news") or {}).get("method") == newsmod.METHOD
        and (read.get("stock_research") or {}).get("as_of") == read.get("as_of")
        and ((read.get("stock_research") or {}).get("source") or {}).get("news_method") == newsmod.METHOD
        and sorted(coverage.get("headline_symbols") or []) == sorted(set(symbols))
        and coverage.get("missing_headline_symbols") == []
        and coverage.get("capped_headline_symbols") == []
        and not diagnostics.get("provider_http_error_symbol_count")
        and all((row.get("provider_response") or {}).get("http_success") is not False
                and (row.get("provider_response") or {}).get("payload_valid") is not False
                for row in (diagnostics.get("symbols") or {}).values())
    )


def _retainable(read, session):
    """Only retain fresh, fully validated evidence, never a legacy false zero."""
    try:
        stock = read["stock_research"]
        coverage = stock["coverage"]
        requested = coverage["requested_symbols"]
        day = read["as_of"]
        close = pd.Timestamp(read["session_close"])
        written, expiry = pd.Timestamp(read["updated_at"]), pd.Timestamp(read["expires_at"])
        now = pd.Timestamp(sessions.utc_now())
        rows = stock["findings"]
        return bool(session and day >= session and read.get("healthy") is True
            and read.get("finalized") is True and read.get("schema_version") == 2
            and read.get("status") == "complete" and read.get("errors") == []
            and not read.get("warnings") and stock.get("schema_version") == findings.SCHEMA_VERSION
            and stock.get("as_of") == day and requested and _complete_news(read, requested)
            and stock["source"]["news_method"] == newsmod.METHOD
            and sorted(coverage["price_symbols"]) == sorted(requested)
            and coverage["missing_price_symbols"] == []
            and len(rows) == len(requested) and {r["symbol"] for r in rows} == set(requested)
            and all(r["id"] == f'{day}:{r["symbol"]}' and r["as_of"] == day
                    and r["price_status"] == r["headline_status"] == "available" for r in rows)
            and all(t.tzinfo is not None for t in (close, written, expiry, now))
            and close == sessions.schedule(day, day).iloc[0].market_close
            and written >= close + pd.Timedelta(minutes=15) and written <= now < expiry
            and expiry == pd.Timestamp(sessions.next_expiry(day)))
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return False


def _retain(read, errors, status, session, failed_sample=None):
    retained = dict(read, last_attempt={"at": now_iso(), "session": session,
                                       "status": status, "errors": errors})
    if failed_sample is not None:
        news = failed_sample.get("news") or {}
        retained["last_attempt"]["news"] = {
            "method": news.get("method"), "failed_symbols": news.get("failed_symbols", []),
            "capped_symbols": news.get("capped_symbols", []),
            "provider_http_error_symbol_count": (news.get("diagnostics") or {}).get("provider_http_error_symbol_count", 0),
        }
    _save(retained)
    report.write_site(retained)
    print(f"WARNING: {status} retry for {session}; retaining validated report dated {read['as_of']}.")


def _publish_failure(errors, status="unavailable", session=None):
    prev = _previous()
    if session is None:
        try:
            session = sessions.last_completed()
        except Exception:
            session = None
    if _retainable(prev, session):
        _retain(prev, errors, status, session)
        return
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
    attempt = {}
    try:
        return _run(args, attempt)
    except Exception as exc:
        _publish_failure([f"Market read failed: {type(exc).__name__}: {exc}"], session=attempt.get("session"))
        return 1


def _run(args, attempt=None):
    cfg = MarketConfig()
    session = sessions.today_session()
    recovery_only = getattr(args, "recovery_only", False)
    if recovery_only or ((session is None or not session["ready"])
                         and getattr(args, "recover_last_completed", False)):
        # Recovery never re-labels yesterday as today. The calendar chooses
        # only a session whose close + settlement buffer has passed.
        day = sessions.last_completed()
        completed = sessions.schedule(day, day)
        session = {"date": day, "close": completed.iloc[0].market_close, "ready": True}
        print(f"Recovering latest completed NYSE session {day}.")
    if session is None or not session["ready"]:
        # Holidays and early manual runs never change the permanent research record.
        prev = _previous()
        if prev.get("finalized") and prev.get("as_of") == sessions.last_completed():
            print("No new completed session; retaining the dated final read.")
            return 0
        _publish_failure(["Waiting for a completed NYSE session and a 15-minute settlement buffer."], "waiting")
        return 0
    as_of = session["date"]
    if attempt is not None:
        attempt["session"] = as_of
    prev = _previous()
    universe = sorted(set(cfg.watchlist) | set(INDEXES) | set(SECTORS) | set(RATES_AND_FEAR))
    complete_daily_prices = (
        not prev.get("price_recovery")
        and (prev.get("universe") or {}).get("loaded") == len(universe)
        and ((prev.get("stock_research") or {}).get("coverage") or {}).get("missing_price_symbols") == []
        and len(prev.get("sectors") or []) == len(SECTORS)
    )
    if ((getattr(args, "scheduled", False) or recovery_only)
            and prev.get("healthy") and prev.get("finalized")
            and prev.get("as_of") == as_of and prev.get("schema_version") == 2
            and (prev.get("stock_research") or {}).get("schema_version") == findings.SCHEMA_VERSION
            and _complete_news(prev, cfg.watchlist)
            and (complete_daily_prices if recovery_only else not prev.get("warnings"))):
        print(f"Completed healthy read already recorded for {as_of}; retry skipped.")
        return 0
    print(f"Loading {len(universe)} symbols for completed session {as_of}...")
    bars = datamod.load_session(universe, start=cfg.history_start,
                                session=as_of, cached=args.cached)
    bars = {sym: df.loc[df.index <= as_of] for sym, df in bars.items()}
    stale = [sym for sym, df in bars.items() if df.empty or str(df.index[-1].date()) != as_of]
    bars = {sym: df for sym, df in bars.items() if sym not in stale}
    missing = sorted(set(universe) - set(bars))
    errors, warnings = [], []
    price_recovery = {sym: df.attrs["price_recovery"] for sym, df in bars.items()
                      if df.attrs.get("price_recovery", {}).get("as_of") == as_of}
    if price_recovery:
        warnings.append(f"Prices recovered from complete intraday bars for {len(price_recovery)} symbols; "
                        "values may differ from finalized daily prints")
    if missing:
        warnings.append(f"Missing or stale prices for {len(missing)} symbols: {', '.join(missing[:10])}")
    spy = bars.get(cfg.regime_symbol)
    if spy is None or len(spy) < cfg.sma_slow + 1:
        raise ValueError(f"{cfg.regime_symbol} has no complete, current history for {as_of} "
                         f"({0 if spy is None else len(spy)} usable bars; "
                         f"need {cfg.sma_slow + 1})")
    idx = regime.index_read(spy, cfg)
    indexes = {s: regime.index_read(bars[s], cfg) for s in INDEXES if s in bars}
    br = regime.breadth(bars, cfg.watchlist, cfg)
    br["coverage_pct"] = round(br.get("counted", 0) / len(cfg.watchlist) * 100, 1)
    if br["coverage_pct"] < 90:
        errors.append("Breadth coverage below 90%; breadth check is unknown")
    vol = regime.volatility(spy, bars.get("^VIX"), cfg)
    if "vix" not in vol:
        errors.append("VIX unavailable; volatility check is unknown")
    # Sector returns compare the exact SPY start/end sessions. An unrelated
    # missing intermediate bar does not change these endpoint returns.
    secs = regime.sector_strength(bars, cfg)
    if len(secs) != len(SECTORS):
        warnings.append("Some sectors lack prices on comparison endpoint dates")
    risk = regime.risk_score(idx, br if br["coverage_pct"] >= 90 else {}, vol, cfg)

    counts, titles, articles, failed, capped = {}, {}, {}, [], []
    news_diagnostics, data_notes = {}, []
    history = _load_news_history()
    baseline = _news_baseline([h for h in history if h.get("date", "") < as_of],
                              cfg.watchlist, cfg.news_baseline_days)
    if not (args.no_news or args.cached):
        counts, articles, failed, capped = newsmod.fetch_headlines(
            cfg.watchlist, end=session["close"], detailed=True, diagnostics=news_diagnostics)
        # Feed-health observations are report-only data notes, not report
        # warnings: they must not change status, the trader-visible warnings,
        # or the scheduled retry skip for an already-recorded session.
        data_notes.extend(news_diagnostics.get("warnings", []))
        titles = {sym: [h['title'] for h in rows] for sym, rows in articles.items()}
        if failed:
            warnings.append(f"Headlines unavailable for {len(failed)} symbols")
        # Capped responses are omitted, not interpreted as a quiet day or a known total.
        if capped:
            warnings.append(f"Headline sample limit reached for {', '.join(capped)}; excluded from baseline")
        if counts:
            old = next((h for h in history if h.get("date") == as_of and h.get("method") == newsmod.METHOD), {})
            row = {"date": as_of, "method": newsmod.METHOD, "window_end": session["close"].isoformat(),
                   "counts": dict(old.get("counts", {}), **counts)}
            history = sorted([h for h in history if not (h.get("date") == as_of
                and h.get("method") == newsmod.METHOD)] + [row], key=lambda h: h["date"])
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            NEWS_FILE.write_text("".join(json.dumps(h) + "\n" for h in history))
    else:
        warnings.append("Headlines skipped; news events were not assessed")
    watch_bars = {s: bars[s] for s in cfg.watchlist if s in bars}
    today = ev.detect(watch_bars, counts, baseline, titles, cfg, as_of=as_of)
    # Only finalized v2 events contribute to the new research series.
    record = ev.merge(ev.load_events(), today)
    filled = ev.fill_follow_through(record, watch_bars, cfg)
    ev.save_events(record)
    says = ev.what_the_record_says(record, cfg)
    stock_research = findings.build(as_of=as_of, symbols=cfg.watchlist,
        bars=watch_bars, counts=counts, headlines=articles, baseline=baseline,
        events=today, sectors=secs, failed=failed, capped=capped,
        skipped=bool(args.no_news or args.cached), cfg=cfg)
    history_gaps = set(watch_bars) - set(stock_research['coverage']['price_symbols'])
    if history_gaps:
        warnings.append(f'Incomplete recent price history for {len(history_gaps)} symbols; unusual moves not assessed')
    read = {
        "schema_version": 2, "rules_version": 2, "updated_at": now_iso(), "as_of": as_of,
        "session_close": session["close"].isoformat(), "expires_at": sessions.next_expiry(as_of),
        "finalized": True, "healthy": not errors,
        "status": "degraded" if errors else "complete_with_warnings" if warnings else "complete",
        "errors": errors, "warnings": warnings,
        "risk": risk, "index": idx, "indexes": indexes, "breadth": br,
        "stock_research": stock_research,
        "volatility": vol, "sectors": secs, "events_today": today,
        "record": {"events_total": len(record), "with_5d": sum("5d" in (e.get("follow") or {}) for e in record),
                   "filled_this_run": filled, "what_it_says": says},
        "news": {"fetched": bool(counts), "baseline_days": len([h for h in history if h.get("method") == newsmod.METHOD]),
                 "baseline_ready": len(baseline) == len(cfg.watchlist), "ready_symbols": len(baseline),
                 "valid_symbols": len(counts), "failed_symbols": failed, "capped_symbols": capped,
                 "method": newsmod.METHOD, "note": "Returned Yahoo articles in the 24 hours ending at the close; not all published news."},
        "min_sample": cfg.min_sample, "universe": {"watchlist": len(cfg.watchlist), "loaded": len(bars)},
    }
    if news_diagnostics:
        read["news"]["diagnostics"] = news_diagnostics
    if data_notes:
        read["news"]["data_notes"] = data_notes
    read["batch_metrics"], read["news_health"] = _batch_metrics(
        watchlist=cfg.watchlist, bars=bars, price_recovery=price_recovery, articles=articles,
        failed=failed, capped=capped, skipped=bool(args.no_news or args.cached),
        diagnostics=news_diagnostics)
    if price_recovery:
        read["price_recovery"] = price_recovery
    read["news"]["validation_version"] = newsmod.VALIDATION_VERSION
    if errors:
        read["last_good"] = (prev if prev.get("healthy") and prev.get("finalized") else prev.get("last_good"))
    read["brief"] = report.brief(read)
    if errors and _retainable(prev, as_of):
        _retain(prev, errors, read["status"], as_of, failed_sample=read)
        return 1
    _save(read)
    print("\n" + read["brief"] + "\n")
    for message in warnings + data_notes:
        print(f"DATA NOTE: {message}")
    for message in errors:
        print(f"ERROR: {message}")
    print(f"Page: {report.write_site(read)}")
    return 0 if read["healthy"] else 1


def _batch_metrics(*, watchlist, bars, price_recovery, articles, failed, capped, skipped,
                   diagnostics):
    """Report-only batch counts. Nothing here feeds healthy, status, events or baselines."""
    in_window = {sym: len(articles.get(sym) or []) for sym in watchlist}
    total = sum(in_window.values())
    with_articles = sum(1 for n in in_window.values() if n)
    responding = len(watchlist) - len(failed)
    if skipped:
        status = "skipped"
    elif total == 0 and len(watchlist) > 1 and responding > 0:
        status = "zero_universe"
    elif failed or capped:
        status = "partial"
    else:
        status = "ok"
    watch_loaded = [sym for sym in watchlist if sym in bars]
    metrics = {
        "session_article_count": total, "symbols_with_articles": with_articles,
        "headline_symbols_requested": 0 if skipped else len(watchlist),
        "price_bars": {
            "universe": {"intraday_rebuilt": len(price_recovery),
                         "daily": len(bars) - len(price_recovery)},
            "watchlist": {"intraday_rebuilt": sum(1 for s in watch_loaded if s in price_recovery),
                          "daily": sum(1 for s in watch_loaded if s not in price_recovery)},
        },
    }
    health = {
        "status": status, "session_article_count": total, "symbols_with_articles": with_articles,
        "failed_symbols": len(failed), "capped_symbols": len(capped),
        "universe_zero_in_window_observed": bool(diagnostics.get("universe_zero_in_window_observed")),
        "note": ("Report only. Does not change healthy, status, per-symbol headline_status, "
                 "events or baselines; a zero is not evidence of no published news."),
    }
    return metrics, health


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
    r.add_argument("--recover-last-completed", action="store_true",
                   help="repair the latest completed session before today's close, preserving its date")
    r.add_argument("--recovery-only", action="store_true",
                   help="repair latest-session price or headline coverage gaps and recovered prices")
    r.set_defaults(func=cmd_run)
    p = sub.add_parser("page", help="rebuild the page from the saved read")
    p.set_defaults(func=cmd_page)
    c = sub.add_parser("record", help="what the event record says so far")
    c.set_defaults(func=cmd_record)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
