"""
What the market agent watches, and the rules it reads the market by.

This agent trades nothing. Its whole job is to look at the same names the
trading agent can buy, plus the indexes and sectors around them, and write
down what the market is doing and how prices reacted to things that
happened. Every number it publishes is computed from free daily bars and free
headline counts. There is no model reading the news and no opinion anywhere:
where a rule turns numbers into a word like "risk-on", the rule is printed
next to the word.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent


def _watchlist() -> List[str]:
    p = HERE / "watchlist.txt"
    return [s.strip().upper() for s in p.read_text().splitlines() if s.strip()]


# The market's own instruments. Breadth is measured on the watchlist; regime,
# volatility and sector strength on these.
INDEXES: Dict[str, str] = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000",
                           "DIA": "Dow"}
SECTORS: Dict[str, str] = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "Health care",
    "XLE": "Energy", "XLY": "Consumer discretionary", "XLP": "Consumer staples",
    "XLI": "Industrials", "XLB": "Materials", "XLU": "Utilities",
    "XLRE": "Real estate", "XLC": "Communication",
}
RATES_AND_FEAR: Dict[str, str] = {"TLT": "Long bonds", "^VIX": "VIX"}


@dataclass
class MarketConfig:
    watchlist: List[str] = field(default_factory=_watchlist)
    history_start: str = "2018-01-01"

    # --- regime -------------------------------------------------------------
    # The broad index against its own long average, the same reading the
    # trading agent gates new entries on, so the two never disagree.
    regime_symbol: str = "SPY"
    sma_fast: int = 50
    sma_slow: int = 200

    # --- breadth ------------------------------------------------------------
    # Share of the watchlist above its own 200-day average. Under this,
    # "the index is up" is a handful of large names carrying it.
    breadth_weak_below: float = 0.40
    breadth_strong_above: float = 0.65

    # --- volatility ---------------------------------------------------------
    vol_window: int = 20            # sessions of realized volatility
    vix_calm_below: float = 15.0
    vix_stressed_above: float = 25.0

    # --- sectors ------------------------------------------------------------
    # Relative strength is the sector's return minus SPY's over these windows.
    rs_short: int = 5
    rs_long: int = 21

    # --- events -------------------------------------------------------------
    # A day is an event for a symbol when its move is large for that symbol
    # (measured in its own ATR, not in percent) or its headline count is far
    # above its usual. Follow-through is filled in on later runs.
    event_move_atr: float = 2.5
    atr_period: int = 14
    news_spike_multiple: float = 3.0
    news_baseline_days: int = 30
    follow_through_days: List[int] = field(default_factory=lambda: [1, 5])
    # Fewer than this in a bucket and the record reports a count, not a
    # result. Same rule the trading agent's journal uses.
    min_sample: int = 20
