"""A malformed daily feed can recover only from complete, compatible sessions."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from mbot import data, sessions


def daily_history(session="2026-10-02"):
    dates = sessions.schedule("2025-01-02", session).index
    return pd.DataFrame(dict(open=10.0, high=11.0, low=9.0, close=10.0, volume=1300), index=dates)


def intraday_history(anchor="2026-10-01", session="2026-10-02", symbol="SPY"):
    calendar = sessions.schedule(anchor, session)
    pieces = []
    for day, offset in ((anchor, 0), (session, 2)):
        row = calendar.loc[day]
        dates = pd.date_range(row.market_open, row.market_close, freq="30min", inclusive="left")
        pieces.append(pd.DataFrame(dict(Open=10.0 + offset, High=11.0 + offset,
                                       Low=9.0 + offset, Close=10.0 + offset, Volume=100), index=dates))
    frame = pd.concat(pieces).tz_convert("America/New_York")
    frame.columns = pd.MultiIndex.from_product([frame.columns, [symbol]], names=["Price", "Ticker"])
    return frame


class IntradayRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cache = patch.object(data, "CACHE_DIR", Path(self.tmp.name))
        cache.start(); self.addCleanup(cache.stop)
        delay = patch.object(data.time, "sleep")
        delay.start(); self.addCleanup(delay.stop)
        self.session = "2026-10-02"
        self.history = daily_history().iloc[:-1]
        self.intraday = intraday_history()

    def recover(self, raw=None, history=None, session=None, symbol="SPY"):
        with patch("yfinance.download", return_value=self.intraday if raw is None else raw) as download:
            result = data._recover_session_intraday(symbol, self.history if history is None else history,
                                                   session or self.session)
        return result, download

    def assert_refused(self, raw, message=None):
        with self.assertRaisesRegex(data.DataError, message or ".+"):
            self.recover(raw)

    def test_complete_sessions_recover_with_provenance_and_one_bounded_request(self):
        result, download = self.recover()
        pd.testing.assert_frame_equal(result.iloc[:-1], self.history, check_names=False)
        self.assertEqual(result.loc[self.session].to_dict(),
                         dict(open=12.0, high=13.0, low=11.0, close=12.0, volume=1300.0))
        self.assertEqual(result.attrs["price_recovery"], {
            "provider": "Yahoo Finance", "as_of": self.session,
            "method": "complete_regular_session_30m", "interval": "30m",
            "anchor_session": "2026-10-01", "bars": 13,
        })
        download.assert_called_once_with("SPY", start="2026-10-01", end="2026-10-03",
                                         interval="30m", auto_adjust=True, prepost=False,
                                         ignore_tz=False, progress=False, threads=False,
                                         group_by="column", timeout=15)
        self.assertFalse(data._cache_path("SPY").exists())

    def test_incident_all_nan_daily_prices_valid_volume_recover_without_caching_reconstruction(self):
        malformed = daily_history()
        malformed.loc[self.session, data.REQUIRED_COLS[:4]] = np.nan
        malformed.loc[self.session, "volume"] = 45_000_000
        with patch.object(data, "download_many", return_value={"SPY": malformed, "^VIX": daily_history()}), \
             patch.object(data, "download_bars", return_value=malformed) as daily, \
             patch("yfinance.download", return_value=self.intraday) as intraday, \
             contextlib.redirect_stdout(io.StringIO()):
            bars = data.load_session(["SPY", "^VIX"], "2025-01-02", self.session)
        self.assertEqual(daily.call_count, 2)
        self.assertEqual(intraday.call_count, 1)
        self.assertEqual(bars["SPY"].loc[self.session, "close"], 12)
        self.assertEqual(bars["SPY"].attrs["price_recovery"]["as_of"], self.session)
        self.assertNotIn("price_recovery", bars["^VIX"].attrs)
        cached = pd.read_csv(data._cache_path("SPY"), index_col=0, parse_dates=True)
        self.assertNotIn(pd.Timestamp(self.session), cached.index)
        self.assertIn(self.history.index[0], cached.index)

    def test_valid_daily_feed_never_calls_intraday(self):
        with patch.object(data, "download_many", return_value={"SPY": daily_history(), "^VIX": daily_history()}), \
             patch.object(data, "_recover_session_intraday") as recover:
            bars = data.load_session(["SPY", "^VIX"], "2025-01-02", self.session)
        recover.assert_not_called()
        self.assertEqual(set(bars), {"SPY", "^VIX"})

    def test_cached_mode_stays_offline_when_session_is_missing(self):
        self.history.to_csv(data._cache_path("SPY"))
        with patch.object(data, "_recover_session_intraday") as recover, \
             patch("yfinance.download", side_effect=AssertionError("network")):
            bars = data.load_session(["SPY"], "2025-01-02", self.session, cached=True)
        recover.assert_not_called()
        self.assertEqual(bars, {})

    def test_missing_slot_in_anchor_or_current_session_is_refused(self):
        for position in (0, 4, 12, 13, 20, 25):
            with self.subTest(position=position):
                self.assert_refused(self.intraday.drop(self.intraday.index[position]), "incomplete")

    def test_stale_or_partial_current_session_is_refused(self):
        for keep in (13, 14, 18, 25):
            with self.subTest(keep=keep):
                self.assert_refused(self.intraday.iloc[:keep], "incomplete")

    def test_duplicate_or_off_grid_regular_slot_is_refused(self):
        duplicate = pd.concat([self.intraday, self.intraday.iloc[[20]]])
        self.assert_refused(duplicate, "duplicate")
        shifted = self.intraday.copy()
        index = shifted.index.to_list()
        index[20] += pd.Timedelta(minutes=1)
        shifted.index = pd.DatetimeIndex(index)
        self.assert_refused(shifted, "incomplete")

    def test_naive_timestamp_and_invalid_datetime_are_refused(self):
        naive = self.intraday.tz_localize(None)
        self.assert_refused(naive, "timezone")
        invalid = self.intraday.copy()
        invalid.index = invalid.index.to_list()[:-1] + [pd.NaT]
        self.assert_refused(invalid, "timezone")

    def test_nonfinite_nonnumeric_and_invalid_ohlcv_slot_are_refused(self):
        for field, invalid in (("Open", np.nan), ("High", np.inf), ("Low", -np.inf),
                               ("Close", "missing"), ("Volume", np.nan), ("Volume", -1),
                               ("Open", 0), ("High", 10), ("Low", 14)):
            with self.subTest(field=field, invalid=invalid):
                malformed = self.intraday.astype(object)
                malformed.iloc[20, malformed.columns.get_loc((field, "SPY"))] = invalid
                self.assert_refused(malformed, "malformed")

    def test_pre_and_post_market_rows_do_not_change_aggregate(self):
        extra = self.intraday.iloc[[0, -1]].astype(float)
        extra.index = pd.DatetimeIndex([self.intraday.index[0] - pd.Timedelta(minutes=30),
                                       self.intraday.index[-1] + pd.Timedelta(minutes=30)])
        extra.loc[:, :] = np.nan
        result, _ = self.recover(pd.concat([extra, self.intraday]).sort_index())
        self.assertEqual(result.loc[self.session, "high"], 13)
        self.assertEqual(result.loc[self.session, "volume"], 1300)

    def test_adjustment_anchor_checks_every_ohlc_field(self):
        for field in data.REQUIRED_COLS[:4]:
            with self.subTest(field=field):
                history = self.history.copy()
                history.loc["2026-10-01", field] *= 1.003
                with self.assertRaisesRegex(data.DataError, "adjustment anchor differs"):
                    self.recover(history=history)

    def test_small_anchor_rounding_difference_is_accepted_without_rescaling(self):
        history = self.history.copy()
        history.loc["2026-10-01", data.REQUIRED_COLS[:4]] *= 1.001
        result, _ = self.recover(history=history)
        self.assertEqual(result.loc[self.session, "close"], 12)

    def test_missing_or_invalid_anchor_is_refused_before_network(self):
        for history in (self.history.iloc[:-1], self.history.iloc[:0]):
            with patch("yfinance.download") as download:
                with self.assertRaisesRegex(data.DataError, "no validated daily adjustment anchor"):
                    data._recover_session_intraday("SPY", history, self.session)
            download.assert_not_called()
        history = self.history.copy()
        history.loc["2026-10-01", "high"] = np.nan
        with patch("yfinance.download") as download:
            with self.assertRaisesRegex(data.DataError, "invalid daily adjustment anchor"):
                data._recover_session_intraday("SPY", history, self.session)
        download.assert_not_called()

    def test_index_session_is_refused_before_network(self):
        with patch("yfinance.download") as download:
            with self.assertRaisesRegex(data.DataError, "does not support index sessions"):
                data._recover_session_intraday("^VIX", self.history, self.session)
        download.assert_not_called()

    def test_early_close_and_holiday_anchor_use_exact_calendar(self):
        session, anchor = "2026-11-27", "2026-11-25"
        result, download = self.recover(intraday_history(anchor, session),
                                       daily_history(session).iloc[:-1], session)
        self.assertEqual(result.attrs["price_recovery"]["anchor_session"], anchor)
        self.assertEqual(result.attrs["price_recovery"]["bars"], 7)
        self.assertEqual(result.loc[session, "volume"], 700)
        self.assertEqual(download.call_args.kwargs["start"], anchor)

    def test_dst_boundary_uses_session_utc_times(self):
        session, anchor = "2026-11-02", "2026-10-30"
        raw = intraday_history(anchor, session).tz_convert("UTC")
        self.assertEqual(raw.index[0].hour, 13)
        self.assertEqual(raw.index[13].hour, 14)
        result, _ = self.recover(raw, daily_history(session).iloc[:-1], session)
        self.assertEqual(result.loc[session, "volume"], 1300)

    def test_download_error_remains_unavailable_without_retry(self):
        with patch("yfinance.download", side_effect=RuntimeError("provider unavailable")) as download:
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                data._recover_session_intraday("SPY", self.history, self.session)
        download.assert_called_once()

    def test_intraday_budget_prioritizes_market_inputs_and_stops(self):
        stale = {sym: self.history for sym in ["AAPL", "SPY", "^VIX", "MSFT"]}
        recovered = daily_history()
        with patch.object(data, "load_universe", return_value=stale), \
             patch.object(data, "load_bars", return_value=self.history), \
             patch.object(data, "download_bars", return_value=self.history), \
             patch.object(data, "_recover_session_intraday", return_value=recovered) as recover, \
             patch.object(data.time, "monotonic", side_effect=[0, 0, 100, 180]), \
             contextlib.redirect_stdout(io.StringIO()) as log:
            bars = data.load_session(list(stale), "2025-01-02", self.session)
        self.assertEqual([call.args[0] for call in recover.call_args_list], ["SPY", "AAPL"])
        self.assertEqual(set(bars), {"SPY", "AAPL"})
        self.assertEqual(log.getvalue().count("time budget exhausted"), 1)


if __name__ == "__main__":
    unittest.main()
