"""
Reading the market from its own prices. Every function here takes bars and
returns numbers with the rule that produced them.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import MarketConfig, SECTORS, INDEXES


def _last(df: pd.DataFrame, col: str = "close") -> Optional[float]:
    try:
        v = float(df[col].iloc[-1])
        return v if v == v else None
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def index_read(df: pd.DataFrame, cfg: MarketConfig) -> dict:
    """Where an index sits against its own averages, and how it has moved."""
    if df is None or len(df) < cfg.sma_slow + 1:
        return {"status": "warming up", "bars": 0 if df is None else len(df)}
    close = df["close"].astype(float)
    fast = close.rolling(cfg.sma_fast).mean()
    slow = close.rolling(cfg.sma_slow).mean()
    last, f, s = float(close.iloc[-1]), float(fast.iloc[-1]), float(slow.iloc[-1])
    above_slow = last > s
    stacked = f > s
    if above_slow and stacked:
        status = "uptrend"
    elif above_slow:
        status = "above the 200-day, 50-day still below it"
    elif stacked:
        status = "below the 200-day, averages still stacked up"
    else:
        status = "downtrend"
    ret = lambda n: (last / float(close.iloc[-1 - n]) - 1) * 100 if len(close) > n else None
    return {
        "close": round(last, 2),
        "sma_fast": round(f, 2), "sma_slow": round(s, 2),
        "above_200": bool(above_slow), "stacked": bool(stacked),
        "status": status,
        "pct_from_200": round((last / s - 1) * 100, 2),
        "ret_1d": round(ret(1), 2) if ret(1) is not None else None,
        "ret_5d": round(ret(5), 2) if ret(5) is not None else None,
        "ret_21d": round(ret(21), 2) if ret(21) is not None else None,
        "high_252": round(float(close.tail(252).max()), 2),
        "pct_from_high": round((last / float(close.tail(252).max()) - 1) * 100, 2),
    }


def breadth(bars: Dict[str, pd.DataFrame], symbols: List[str], cfg: MarketConfig) -> dict:
    """Share of the watchlist above its own 200-day and 50-day averages,
    and how many made a 20-day high or low today."""
    above_slow = above_fast = highs = lows = counted = 0
    for s in symbols:
        df = bars.get(s)
        if df is None or len(df) < cfg.sma_slow + 1:
            continue
        close = df["close"].astype(float)
        last = float(close.iloc[-1])
        counted += 1
        if last > float(close.rolling(cfg.sma_slow).mean().iloc[-1]):
            above_slow += 1
        if last > float(close.rolling(cfg.sma_fast).mean().iloc[-1]):
            above_fast += 1
        window = close.tail(20)
        if last >= float(window.max()):
            highs += 1
        if last <= float(window.min()):
            lows += 1
    if counted == 0:
        return {"counted": 0, "status": "no data"}
    share = above_slow / counted
    if share < cfg.breadth_weak_below:
        status = "weak"
    elif share > cfg.breadth_strong_above:
        status = "broad"
    else:
        status = "mixed"
    return {
        "counted": counted,
        "above_200_pct": round(share * 100, 1),
        "above_50_pct": round(above_fast / counted * 100, 1),
        "new_20d_highs": highs, "new_20d_lows": lows,
        "status": status,
        "rule": f"weak under {cfg.breadth_weak_below:.0%} above the 200-day, "
                f"broad over {cfg.breadth_strong_above:.0%}",
    }


def volatility(spy: pd.DataFrame, vix: Optional[pd.DataFrame], cfg: MarketConfig) -> dict:
    """Realized volatility of the index and the level of the VIX."""
    out = {}
    if spy is not None and len(spy) > cfg.vol_window + 1:
        r = spy["close"].astype(float).pct_change().dropna()
        rv = float(r.tail(cfg.vol_window).std() * math.sqrt(252) * 100)
        rv_prev = float(r.tail(cfg.vol_window * 2).head(cfg.vol_window).std() * math.sqrt(252) * 100)
        out["realized_20d_pct"] = round(rv, 1)
        out["realized_prev_20d_pct"] = round(rv_prev, 1)
        out["realized_trend"] = "rising" if rv > rv_prev * 1.15 else ("falling" if rv < rv_prev * 0.85 else "steady")
    if vix is not None and len(vix) > 5:
        v = float(vix["close"].iloc[-1])
        v5 = float(vix["close"].iloc[-6])
        out["vix"] = round(v, 2)
        out["vix_change_5d_pct"] = round((v / v5 - 1) * 100, 1)
        out["vix_status"] = ("calm" if v < cfg.vix_calm_below
                             else "stressed" if v > cfg.vix_stressed_above else "normal")
        out["vix_rule"] = f"calm under {cfg.vix_calm_below:g}, stressed over {cfg.vix_stressed_above:g}"
    return out


def sector_strength(bars: Dict[str, pd.DataFrame], cfg: MarketConfig) -> List[dict]:
    """Each sector's return minus SPY's over a short and a long window,
    ranked by the long one. Leadership is what the index is made of."""
    spy = bars.get("SPY")
    if spy is None or len(spy) < cfg.rs_long + 2:
        return []
    def ret(df, n):
        c = df["close"].astype(float)
        return float(c.iloc[-1] / c.iloc[-1 - n] - 1) * 100 if len(c) > n else None
    spy_s, spy_l = ret(spy, cfg.rs_short), ret(spy, cfg.rs_long)
    rows = []
    for sym, name in SECTORS.items():
        df = bars.get(sym)
        if df is None or len(df) < cfg.rs_long + 2:
            continue
        s, l = ret(df, cfg.rs_short), ret(df, cfg.rs_long)
        if s is None or l is None:
            continue
        rows.append({"symbol": sym, "name": name,
                     "ret_5d": round(s, 2), "ret_21d": round(l, 2),
                     "rs_5d": round(s - spy_s, 2), "rs_21d": round(l - spy_l, 2)})
    rows.sort(key=lambda r: -r["rs_21d"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def risk_score(idx: dict, br: dict, vol: dict, cfg: MarketConfig) -> dict:
    """One word for the weather, and the four checks that produced it.

    Four yes/no checks, each documented. Three or four yes is risk-on, zero
    or one is risk-off, two is mixed. It is a summary of the numbers above
    it, not a forecast, and it should be read as "what the market is doing",
    never as "what it will do".
    """
    checks = [
        ("index above its 200-day", bool(idx.get("above_200"))),
        ("50-day above the 200-day", bool(idx.get("stacked"))),
        (f"breadth not weak (over {cfg.breadth_weak_below:.0%} of names above their 200-day)",
         br.get("status") in ("mixed", "broad")),
        (f"VIX not stressed (under {cfg.vix_stressed_above:g})",
         vol.get("vix_status") in ("calm", "normal") if "vix_status" in vol else True),
    ]
    yes = sum(1 for _, ok in checks if ok)
    word = "risk-on" if yes >= 3 else ("risk-off" if yes <= 1 else "mixed")
    return {"word": word, "score": yes, "of": len(checks),
            "checks": [{"check": c, "ok": ok} for c, ok in checks]}
