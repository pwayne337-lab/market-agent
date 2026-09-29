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
             patch.object(data, 'download_bars', return_value=self.old) as download:
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
             patch.object(data,'load_bars',side_effect=data.DataError('provider down')):
            self.assertEqual(data.load_session(['SPY'],'2026-01-01','2026-09-28'), {})
