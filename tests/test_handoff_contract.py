"""Lock the actual report writer to the offline fixture consumed by the trader.

The identical JSON fixture lives in each repository, so CI needs no checkout
of the other project. It contains synthetic observations, never broker data.
"""
import argparse
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import agent
from mbot import events, news, report, sessions
from mbot.config import INDEXES, RATES_AND_FEAR, SECTORS, MarketConfig


FIXTURE = Path(__file__).parent / "fixtures" / "researcher_report_v1.json"
WRITTEN_AT = "2026-09-24T20:47:00+00:00"


def generate_report():
    """Run the public producer command with real calculations and mocked feeds."""
    cfg = MarketConfig(watchlist=["AAPL", "MSFT", "NVDA"])
    dates = sessions.schedule("2025-01-01", "2026-09-24").index
    close = np.linspace(100, 150, len(dates))
    frame = pd.DataFrame(
        dict(open=close, high=close + 1, low=close - 1, close=close, volume=1000),
        index=dates,
    )
    symbols = set(cfg.watchlist) | set(INDEXES) | set(SECTORS) | set(RATES_AND_FEAR)
    bars = {symbol: frame.copy() for symbol in sorted(symbols)}
    bars["^VIX"].loc[:, ["open", "high", "low", "close"]] = [20, 21, 19, 20]
    # A valid large final bar, sufficient to trigger the real ATR event rule.
    bars["AAPL"].loc[dates[-1], ["open", "high", "low", "close"]] = [160, 163, 159, 162]
    articles = {
        "AAPL": [
            {"title": "Fixture: AAPL announces a product update", "publisher": "Fixture Wire",
             "published_at": "2026-09-24T18:00:00+00:00", "source": "Yahoo Finance",
             "url": "https://example.com/fixture/aapl-product-update"},
            # The producer's 24-hour window also includes yesterday after close.
            {"title": "Fixture: AAPL previews its product event", "publisher": "Fixture Wire",
             "published_at": "2026-09-23T21:00:00+00:00", "source": "Yahoo Finance",
             "url": "https://example.com/fixture/aapl-event-preview"},
        ],
        "MSFT": [],
    }
    args = argparse.Namespace(cached=False, no_news=False, scheduled=False,
                              recover_last_completed=False)
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        state = root / "state"
        state.mkdir()
        paths = [(agent, "STATE_DIR", state), (agent, "READ_FILE", state / "market.json"),
                 (agent, "NEWS_FILE", state / "news_counts.jsonl"),
                 (events, "STATE_DIR", state), (events, "EVENTS_FILE", state / "events.jsonl"),
                 (report, "SITE", root / "site")]
        for module, name, value in paths:
            stack.enter_context(patch.object(module, name, value))
        # Ready, valid history makes a quiet stock distinguishable from missing data.
        history = [{"date": day, "method": news.METHOD,
                    "counts": {symbol: 1 for symbol in cfg.watchlist}}
                   for day in ("2026-09-17", "2026-09-18", "2026-09-21", "2026-09-22", "2026-09-23")]
        agent.NEWS_FILE.write_text("".join(json.dumps(row) + "\n" for row in history))
        stack.enter_context(patch.object(agent, "MarketConfig", return_value=cfg))
        stack.enter_context(patch.object(agent, "now_iso", return_value=WRITTEN_AT))
        stack.enter_context(patch.object(sessions, "utc_now", return_value=pd.Timestamp(WRITTEN_AT)))
        prices = stack.enter_context(patch.object(agent.datamod, "load_session", return_value=bars))
        headlines = stack.enter_context(patch.object(news, "fetch_headlines",
            return_value=({"AAPL": 2, "MSFT": 0}, articles, ["NVDA"], [])))
        stack.enter_context(redirect_stdout(io.StringIO()))
        result = agent.cmd_run(args)
        if result != 0:
            raise AssertionError(agent.READ_FILE.read_text())
        prices.assert_called_once_with(sorted(symbols), start=cfg.history_start,
                                       session="2026-09-24", cached=False)
        headlines.assert_called_once_with(cfg.watchlist, end=pd.Timestamp("2026-09-24T20:00Z"),
                                          detailed=True)
        saved = json.loads(agent.READ_FILE.read_text())
        published = json.loads((report.SITE / "market.json").read_text())
        if saved != published:
            raise AssertionError("The published handoff differs from the saved research record")
        return published


class HandoffContractTests(unittest.TestCase):
    def test_real_report_writer_matches_shared_contract(self):
        actual = generate_report()
        expected = json.loads(FIXTURE.read_text())
        # Brief wording and record presentation can evolve independently of the
        # machine contract; freshness, health and every finding are locked here.
        fields = ("schema_version", "rules_version", "updated_at", "as_of", "session_close",
                  "expires_at", "finalized", "healthy", "status", "errors", "warnings",
                  "stock_research")
        self.assertEqual({k: actual[k] for k in fields}, {k: expected[k] for k in fields})
        rows = {row["symbol"]: row for row in actual["stock_research"]["findings"]}
        self.assertEqual(rows["AAPL"]["events"][0]["kind"], "move")
        self.assertEqual(len(rows["AAPL"]["headlines"]), 2)
        self.assertEqual(rows["MSFT"]["events"], [])
        self.assertEqual(rows["MSFT"]["headlines"], [])
        self.assertEqual(rows["MSFT"]["news_baseline_status"], "ready")
        self.assertEqual(rows["NVDA"]["headline_status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
