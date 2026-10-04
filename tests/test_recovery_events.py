"""Reconstructed prices must not become permanent event-learning facts."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import pandas as pd

from mbot import events, sessions
from mbot.config import MarketConfig


class RecoveryEvents(unittest.TestCase):
    def setUp(self):
        self.day = "2026-10-02"
        self.cfg = MarketConfig(watchlist=["AAA"])
        index = sessions.schedule("2026-08-01", self.day).index
        self.daily = pd.DataFrame({"open": 100., "high": 101., "low": 99.,
                                   "close": 100., "volume": 1000.}, index=index)
        self.daily.loc[self.day, ["high", "close"]] = [106., 105.8]
        self.recovered = self.daily.copy()
        self.recovered.loc[self.day, "close"] = 106.
        self.recovered.attrs["price_recovery"] = {
            "as_of": self.day, "method": "complete_regular_session_30m"}

    def detect(self, frame):
        return events.detect({"AAA": frame}, {}, {}, {}, self.cfg, as_of=self.day)

    def test_provisional_threshold_event_disappears_without_entering_record(self):
        observed = self.detect(self.recovered)
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0]["price_recovery"], self.recovered.attrs["price_recovery"])
        record = events.merge([], observed)
        self.assertEqual(record, [])
        final = self.detect(self.daily)
        self.assertEqual(final, [])
        self.assertEqual(events.merge(record, final), [])
        self.assertEqual(events.what_the_record_says(record, self.cfg), [])
        self.recovered.attrs["price_recovery"]["as_of"] = "changed"
        self.assertEqual(observed[0]["price_recovery"]["as_of"], self.day)

    def test_daily_confirmation_enters_record_and_recovered_retry_preserves_it(self):
        confirmed = self.recovered.copy()
        confirmed.attrs.clear()
        final = self.detect(confirmed)
        self.assertEqual(len(final), 1)
        self.assertNotIn("price_recovery", final[0])
        record = events.merge([], final)
        record[0]["follow"] = {"1d": 1.2}
        before = deepcopy(record)
        self.assertEqual(events.merge(record, self.detect(self.recovered)), before)

    def test_stale_recovery_metadata_does_not_mark_daily_event(self):
        self.recovered.attrs["price_recovery"]["as_of"] = "2026-10-01"
        observed = self.detect(self.recovered)
        self.assertNotIn("price_recovery", observed[0])
        self.assertEqual(len(events.merge([], observed)), 1)

    def test_recovered_follow_endpoint_waits_for_daily_confirmation(self):
        event = {"date": "2026-09-25", "symbol": "AAA", "follow": {}}
        with patch.object(events, "last_completed", return_value=self.day):
            self.assertEqual(events.fill_follow_through([event], {"AAA": self.recovered}, self.cfg), 1)
        self.assertEqual(event["follow"], {"1d": 0.0})
        with patch.object(events, "last_completed", return_value=self.day):
            self.assertEqual(events.fill_follow_through([event], {"AAA": self.daily}, self.cfg), 1)
        self.assertEqual(event["follow"], {"1d": 0.0, "5d": 5.8})

    def test_recovered_endpoint_cannot_overwrite_existing_daily_return(self):
        event = {"date": "2026-09-25", "symbol": "AAA", "follow": {"1d": 0., "5d": 5.8}}
        with patch.object(events, "last_completed", return_value=self.day):
            self.assertEqual(events.fill_follow_through([event], {"AAA": self.recovered}, self.cfg), 0)
        self.assertEqual(event["follow"]["5d"], 5.8)

    def test_recovered_formation_endpoint_is_never_used_for_returns(self):
        dates = sessions.schedule(self.day, "2026-10-05").index
        bars = pd.DataFrame({"close": [106., 110.]}, index=dates)
        bars.attrs["price_recovery"] = deepcopy(self.recovered.attrs["price_recovery"])
        event = {"date": self.day, "symbol": "AAA", "follow": {}}
        with patch.object(events, "last_completed", return_value="2026-10-05"):
            self.assertEqual(events.fill_follow_through([event], {"AAA": bars}, self.cfg), 0)
        self.assertEqual(event["follow"], {})


if __name__ == "__main__":
    unittest.main()
