"""A nonempty stale download must not look like a completed market read."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from mbot import data


def frame(dates):
    return pd.DataFrame(dict(open=10, high=11, low=9, close=10, volume=100),
                        index=pd.to_datetime(dates))


class SessionDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = patch.object(data, 'CACHE_DIR', Path(self.tmp.name))
        p.start(); self.addCleanup(p.stop)
        self.old = frame(['2026-09-24', '2026-09-25'])
        self.fresh = frame(['2026-09-24', '2026-09-25', '2026-09-28'])

    def test_stale_nonempty_batch_is_retried_and_recovered(self):
        with patch.object(data, 'download_many', return_value={'SPY':self.old, '^VIX':self.fresh}), \
             patch.object(data, 'download_bars', return_value=self.fresh) as download:
            bars = data.load_session(['SPY','^VIX'], '2026-01-01', '2026-09-28')
        self.assertEqual(str(bars['SPY'].index[-1].date()), '2026-09-28')
        download.assert_called_once_with('SPY', start='2026-01-01', end='2026-09-29')

    def test_stale_retries_are_refused_and_current_symbols_survive(self):
        with patch.object(data, 'download_many', return_value={'SPY':self.old, '^VIX':self.fresh}), \
             patch.object(data, 'download_bars', return_value=self.old) as download, \
             patch.object(data.time, 'sleep'):
            bars = data.load_session(['SPY','^VIX'], '2026-01-01', '2026-09-28')
        self.assertNotIn('SPY', bars)
        self.assertIn('^VIX', bars)
        self.assertEqual(download.call_count, 2)

    def test_cached_is_offline_and_future_bars_are_trimmed(self):
        self.old.to_csv(data._cache_path('SPY'))
        frame(['2026-09-28','2026-09-29']).to_csv(data._cache_path('^VIX'))
        with patch.object(data,'download_bars',side_effect=AssertionError('network')), \
             patch.object(data,'download_many',side_effect=AssertionError('network')):
            bars = data.load_session(['SPY','^VIX','MISSING'], '2026-01-01', '2026-09-28',cached=True)
        self.assertEqual(list(bars), ['^VIX'])
        self.assertEqual(list(bars['^VIX'].index), [pd.Timestamp('2026-09-28')])

    def test_retry_error_does_not_turn_stale_data_into_success(self):
        with patch.object(data,'load_universe',return_value={'SPY':self.old}), \
             patch.object(data,'load_bars',side_effect=data.DataError('provider down')), \
             patch.object(data,'download_bars',side_effect=data.DataError('provider down')), \
             patch.object(data.time,'sleep'):
            self.assertEqual(data.load_session(['SPY'],'2026-01-01','2026-09-28'), {})

    def test_malformed_latest_bar_recovers_with_different_request_window(self):
        malformed = self.fresh.copy()
        malformed.loc[pd.Timestamp('2026-09-28'), 'high'] = 8
        with patch.object(data, 'download_many', return_value={'SPY': malformed, '^VIX': self.fresh}), \
             patch.object(data, 'download_bars', side_effect=[malformed, self.fresh]) as download, \
             patch.object(data.time, 'sleep') as sleep:
            bars = data.load_session(['SPY', '^VIX'], '2024-01-01', '2026-09-28')
        self.assertIn(pd.Timestamp('2026-09-28'), bars['SPY'].index)
        self.assertEqual(download.call_count, 2)
        self.assertNotEqual(download.call_args_list[0].kwargs['start'], download.call_args_list[1].kwargs['start'])
        sleep.assert_called_once_with(5)

    def test_bad_recovery_bar_remains_unavailable(self):
        malformed = self.fresh.copy()
        malformed.loc[pd.Timestamp('2026-09-28'), 'volume'] = float('nan')
        with patch.object(data, 'download_many', return_value={'SPY': malformed, '^VIX': self.fresh}), \
             patch.object(data, 'download_bars', return_value=malformed), \
             patch.object(data.time, 'sleep'):
            bars = data.load_session(['SPY', '^VIX'], '2024-01-01', '2026-09-28')
        self.assertNotIn('SPY', bars)
        self.assertIn('^VIX', bars)

    def test_nonfinite_and_nonnumeric_ohlcv_are_rejected(self):
        for column in data.REQUIRED_COLS:
            for invalid in (float('inf'), float('-inf'), float('nan'), 'unavailable'):
                with self.subTest(column=column, invalid=invalid):
                    bad = self.fresh.astype(object)
                    bad.loc[pd.Timestamp('2026-09-28'), column] = invalid
                    cleaned = data._sanity_check('SPY', bad)
                    self.assertEqual(list(cleaned.index), list(self.old.index))

    def test_rejected_values_are_logged(self):
        bad = self.fresh.copy()
        bad.loc[pd.Timestamp('2026-09-28'), 'high'] = 8
        with patch('builtins.print') as log:
            data._sanity_check('SPY', bad)
        self.assertIn("'high': 8", log.call_args.args[0])
        self.assertIn('2026-09-28', log.call_args.args[0])

    def test_wholly_malformed_batch_can_recover_and_waits_once(self):
        malformed = frame(['2026-09-28'])
        malformed['high'] = 8
        with patch.object(data, 'download_many', return_value={'SPY': malformed, '^VIX': malformed}), \
             patch.object(data, 'download_bars', side_effect=[malformed, self.fresh, malformed, self.fresh]), \
             patch.object(data.time, 'sleep') as sleep:
            bars = data.load_session(['SPY', '^VIX'], '2024-01-01', '2026-09-28')
        self.assertEqual(set(bars), {'SPY', '^VIX'})
        sleep.assert_called_once_with(5)

    def test_empty_cache_remains_offline_and_unavailable(self):
        with patch.object(data, 'download_bars', side_effect=AssertionError('network')), \
             patch.object(data, 'download_many', side_effect=AssertionError('network')):
            self.assertEqual(data.load_session(['SPY'], '2026-01-01', '2026-09-28', cached=True), {})

    def test_holidays_and_weekends_do_not_enter_session_history(self):
        mixed = frame(['2026-09-07', '2026-09-25', '2026-09-26', '2026-09-27', '2026-09-28'])
        mixed.to_csv(data._cache_path('SPY'))
        bars = data.load_session(['SPY'], '2026-01-01', '2026-09-28', cached=True)
        self.assertEqual(list(bars['SPY'].index), list(pd.to_datetime(['2026-09-25', '2026-09-28'])))

    def test_nontrading_target_is_never_current(self):
        with patch.object(data, 'load_universe') as load:
            with self.assertRaisesRegex(data.DataError, 'not a trading session'):
                data.load_session(['SPY'], '2026-01-01', '2026-09-27')
        load.assert_not_called()

    def test_recovery_window_differs_even_with_short_requested_history(self):
        with patch.object(data, 'download_many', return_value={'SPY': self.old, '^VIX': self.fresh}), \
             patch.object(data, 'download_bars', side_effect=[self.old, self.fresh]) as download, \
             patch.object(data.time, 'sleep'):
            bars = data.load_session(['SPY', '^VIX'], '2026-09-01', '2026-09-28')
        self.assertIn('SPY', bars)
        self.assertEqual(download.call_args.kwargs['start'], '2026-08-25')

    def test_recovery_keeps_compatible_older_history(self):
        history = frame(['2020-01-02', '2026-09-24', '2026-09-25'])
        history.to_csv(data._cache_path('SPY'))
        with patch.object(data, 'download_many', return_value={'SPY': self.old, '^VIX': self.fresh}), \
             patch.object(data, 'download_bars', side_effect=[self.old, self.fresh]), \
             patch.object(data.time, 'sleep'):
            bars = data.load_session(['SPY', '^VIX'], '2020-01-01', '2026-09-28')
        self.assertIn(pd.Timestamp('2020-01-02'), bars['SPY'].index)
        self.assertIn(pd.Timestamp('2026-09-28'), bars['SPY'].index)

    def test_shifted_recovery_rebuilds_incompatible_adjustment_history(self):
        history = frame(['2020-01-02', '2026-09-24', '2026-09-25'])
        history[data.REQUIRED_COLS[:4]] *= 2
        history.to_csv(data._cache_path('SPY'))
        rebuilt = frame(['2020-01-02', '2026-09-24', '2026-09-25', '2026-09-28'])
        with patch.object(data, 'download_bars', return_value=rebuilt) as download:
            result = data.load_bars('SPY', start='2020-01-01', end='2026-09-29',
                                    refresh=True, fresh=self.fresh)
        download.assert_called_once_with('SPY', start='2020-01-01', end=None)
        self.assertTrue((result['close'] == 10).all())
        self.assertEqual(len(result), 4)

    def test_invalid_overlap_is_not_adjustment_compatibility(self):
        bad = self.old.copy()
        bad['close'] = float('inf')
        drift, shared = data._adjustment_drift(bad, bad)
        self.assertEqual(shared, 0)
        self.assertEqual(drift, float('inf'))

    def test_malformed_download_cannot_overwrite_good_cached_bar(self):
        self.fresh.to_csv(data._cache_path('SPY'))
        bad = self.fresh.copy()
        bad.loc[pd.Timestamp('2026-09-28'), 'high'] = 8
        result = data.load_bars('SPY', start='2026-01-01', refresh=True, fresh=bad)
        self.assertEqual(result.loc[pd.Timestamp('2026-09-28'), 'high'], 11)
        cached = pd.read_csv(data._cache_path('SPY'), index_col=0, parse_dates=True)
        self.assertEqual(cached.loc[pd.Timestamp('2026-09-28'), 'high'], 11)

    def test_wholly_malformed_rebuild_preserves_previous_cache(self):
        self.old.to_csv(data._cache_path('SPY'))
        before = data._cache_path('SPY').read_bytes()
        bad = self.fresh.copy()
        bad['high'] = 8
        with patch.object(data, 'download_bars', return_value=bad):
            with self.assertRaisesRegex(data.DataError, 'no usable downloaded bars'):
                data.load_bars('SPY', start='2026-01-01', refresh=True, fresh=bad)
        self.assertEqual(data._cache_path('SPY').read_bytes(), before)
