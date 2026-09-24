"""Offline checks for the market agent. Every number is produced from
synthetic bars, so nothing here depends on a network."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mbot import events as ev, regime, report
from mbot.config import MarketConfig

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f"  -> {detail}"))
    if not ok:
        FAILURES.append(name)


def series(n=400, seed=1, drift=0.0005, vol=0.012, start=100.0):
    rng = np.random.default_rng(seed)
    closes = start * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    opens = np.r_[start, closes[:-1]] * (1 + rng.normal(0, vol * 0.3, n))
    span = np.abs(rng.normal(0, vol, n)) * closes
    highs = np.maximum(opens, closes) + span
    lows = np.minimum(opens, closes) - span
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                         "volume": rng.integers(1e6, 5e6, n)}, index=idx)


cfg = MarketConfig()

print("\nRegime")
up = series(drift=0.002)
down = up.copy(); down["close"] = down["close"].iloc[::-1].to_numpy()
r_up, r_down = regime.index_read(up, cfg), regime.index_read(down, cfg)
check("a rising index reads as an uptrend", r_up["status"] == "uptrend", r_up["status"])
check("a falling one reads as a downtrend", r_down["status"] == "downtrend", r_down["status"])
check("too little history says so rather than guessing",
      regime.index_read(up.head(100), cfg)["status"] == "warming up")

print("\nBreadth")
bars = {f"S{i}": series(seed=i, drift=0.002) for i in range(10)}
bars.update({f"W{i}": down.copy() for i in range(10)})
br = regime.breadth(bars, list(bars), cfg)
check("counts every name with enough history", br["counted"] == 20, br)
check("half above their 200-day reads as mixed", br["status"] == "mixed" and 40 <= br["above_200_pct"] <= 65, br)
br2 = regime.breadth({k: v for k, v in bars.items() if k.startswith("S")}, [k for k in bars if k.startswith("S")], cfg)
check("all above reads as broad", br2["status"] == "broad", br2)

print("\nRisk score")
vol = {"vix_status": "calm"}
rs = regime.risk_score(r_up, br2, vol, cfg)
check("uptrend, stacked, broad, calm is risk-on 4 of 4", rs["word"] == "risk-on" and rs["score"] == 4, rs)
rs2 = regime.risk_score(r_down, {"status": "weak"}, {"vix_status": "stressed"}, cfg)
check("downtrend, weak, stressed is risk-off", rs2["word"] == "risk-off" and rs2["score"] == 0, rs2)
check("every check is printed with its answer", len(rs["checks"]) == 4 and all("check" in c for c in rs["checks"]))
rs3 = regime.risk_score(r_up, br2, {}, cfg)
check("no VIX data does not count against the market", rs3["score"] == 4, rs3)

print("\nEvents")
big = series(seed=7)
big.iloc[-1, big.columns.get_loc("close")] = float(big["close"].iloc[-2]) * 1.08   # an 8% day
quiet = series(seed=8)
today = str(big.index[-1].date())
evs = ev.detect({"BIG": big, "QUIET": quiet}, {"BIG": 9, "QUIET": 1}, {"BIG": 2.0, "QUIET": 2.0},
                {"BIG": ["a", "b", "c", "d"]}, cfg, as_of=today)
check("an 8% day on 4.5x usual headlines is a move+news event",
      [e["symbol"] for e in evs] == ["BIG"] and evs[0]["kind"] == "move+news", str(evs))
check("its direction and size are recorded in ATRs",
      evs[0]["direction"] == "up" and evs[0]["move_atr"] > 2.5, str(evs[0]))
check("only three headlines are kept", len(evs[0]["top_headlines"]) == 3)
evs2 = ev.detect({"BIG": big}, {}, {}, {}, cfg, as_of=today)
check("with no news baseline a big move is still an event, of kind move",
      evs2 and evs2[0]["kind"] == "move", str(evs2))
evs3 = ev.detect({"QUIET": quiet}, {"QUIET": 9}, {"QUIET": 2.0}, {}, cfg, as_of=today)
check("a headline burst with no move is an event of kind news",
      evs3 and evs3[0]["kind"] == "news", str(evs3))
check("a stale bar is not today's event", ev.detect({"BIG": big}, {}, {}, {}, cfg, as_of="2000-01-01") == [])

merged = ev.merge([{"date": today, "symbol": "BIG"}], evs)
check("the same (date, symbol) is not recorded twice", len(merged) == 1)

# Follow-through is filled only once the sessions have happened.
old = {"date": str(big.index[-10].date()), "symbol": "BIG", "kind": "move", "direction": "up", "follow": {}}
n = ev.fill_follow_through([old], {"BIG": big}, cfg)
base = float(big["close"].iloc[-10])
check("one- and five-day follow-through are written from the bars",
      n == 2 and abs(old["follow"]["1d"] - (float(big["close"].iloc[-9]) / base - 1) * 100) < 0.01
      and "5d" in old["follow"], str(old))
fresh = {"date": today, "symbol": "BIG", "kind": "move", "direction": "up", "follow": {}}
check("an event from today has no follow-through yet", ev.fill_follow_through([fresh], {"BIG": big}, cfg) == 0 and fresh["follow"] == {})
check("filling twice does not overwrite", ev.fill_follow_through([old], {"BIG": big}, cfg) == 0)

print("\nWhat the record says")
few = [{"kind": "move", "direction": "up", "follow": {"1d": 1.0, "5d": 2.0}} for _ in range(5)]
says = ev.what_the_record_says(few, cfg)
check("five events report a count and no result",
      says[0]["events"] == 5 and "note" in says[0]["after_1d"] and "mean_pct" not in says[0]["after_1d"], str(says))
many = [{"kind": "move", "direction": "down", "follow": {"1d": -0.5 + (i % 3) * 0.1, "5d": 1.0}} for i in range(30)]
says2 = ev.what_the_record_says(many, cfg)
check("thirty events report a mean, an up-share and a confidence interval",
      says2[0]["after_1d"]["n"] == 30 and "mean_pct" in says2[0]["after_1d"] and "ci" in says2[0]["after_1d"], str(says2))
check("thirty identical +1% five-day results are called real",
      says2[0]["after_5d"].get("real") is True, str(says2[0]["after_5d"]))

print("\nBrief and page")
read = {"as_of": today, "updated_at": "now", "risk": rs, "index": r_up, "breadth": br2,
        "volatility": {"vix": 14.0, "vix_status": "calm", "vix_change_5d_pct": -3.0,
                       "realized_20d_pct": 12.0, "realized_trend": "steady"},
        "sectors": [], "events_today": evs, "record": {"events_total": 1, "with_5d": 0, "what_it_says": says},
        "min_sample": 20}
b = report.brief(read)
check("the brief names the word and the checks", "RISK-ON" in b and "4 of 4" in b, b)
check("and the event", "BIG" in b and "+8.0%" in b, b)
html = report.page(dict(read, brief=b))
check("the page renders the verdict and the event", "risk-on" in html and "BIG" in html)
check("the page refuses to draw a conclusion from five events", "not enough data (5 of 20)" in html)
check("nothing on the page claims to forecast", "forecast" in html and "will " not in b.lower())

print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
