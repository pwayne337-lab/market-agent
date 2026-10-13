"""Required-input failures stay current; earlier evidence is an archive."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import patch

import pandas as pd

import agent
from mbot import news, report, sessions
from tests import test_reliability


class ReportRetention(unittest.TestCase):
    def setUp(self):
        self.fixture = test_reliability.Reliability()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        clock = patch.object(agent, 'now_iso', return_value='2026-09-24T20:47:00+00:00')
        clock.start()
        self.addCleanup(clock.stop)
        _, self.healthy = self.fixture.run_agent(
            news_result=({s:0 for s in self.fixture.cfg.watchlist}, {}, [], []))

    def test_mandatory_price_failures_remain_current_and_archive_previous_evidence(self):
        for missing in ('SPY', '^VIX'):
            with self.subTest(missing=missing):
                bars = {s:f for s,f in self.fixture.bars.items() if s != missing}
                code, current = self.fixture.run_agent(bars)
                self.assertEqual(code, 1)
                self.assertFalse(current['healthy'])
                self.assertTrue(current['errors'])
                self.assertEqual(current['last_good']['report_file'], 'last-good.json')
                self.assertNotIn('stock_research', current['last_good'])
                archived = json.loads((report.SITE/'last-good.json').read_text())
                self.assertEqual(archived, self.healthy)
                published = json.loads((report.SITE/'market.json').read_text())
                self.assertFalse(published['healthy'])
                self.assertIn('Archived previous report', (report.SITE/'index.html').read_text())

    def test_invalid_previous_evidence_is_never_retained_as_current(self):
        for invalid in ('legacy', 'observed_failure', 'expired', 'partial', 'malformed'):
            with self.subTest(invalid=invalid):
                old = deepcopy(self.healthy)
                if invalid == 'legacy':
                    old['news'].pop('validation_version')
                    old['news']['method'] = 'yahoo_returned_items_v2'
                elif invalid == 'observed_failure':
                    old['news']['diagnostics'] = {'provider_http_error_symbol_count':1}
                elif invalid == 'expired':
                    old['expires_at'] = '2026-09-24T20:46:00+00:00'
                elif invalid == 'partial':
                    old['stock_research']['coverage']['missing_price_symbols'] = ['AAPL']
                else:
                    old['stock_research'] = []
                agent._save(old)
                code, current = self.fixture.run_agent({s:f for s,f in self.fixture.bars.items() if s != 'SPY'})
                self.assertEqual(code, 1)
                self.assertFalse(current['healthy'])
                self.assertFalse(current['finalized'])

    def test_later_failed_news_sample_is_not_masked_by_previous_good_report(self):
        bars = {s:f for s,f in self.fixture.bars.items() if s != '^VIX'}
        code, current = self.fixture.run_agent(bars, news_result=({}, {}, ['AAPL'], []))
        self.assertEqual(code, 1)
        self.assertFalse(current['healthy'])
        self.assertEqual(current['news']['failed_symbols'], ['AAPL'])
        row = next(r for r in current['stock_research']['findings'] if r['symbol'] == 'AAPL')
        self.assertEqual(row['headline_status'], 'unavailable')
        self.assertEqual(json.loads((report.SITE/'last-good.json').read_text()), self.healthy)
        self.assertNotIn('stock_research', current['last_good'])

    def test_failure_before_data_collection_can_keep_proven_fresh_report(self):
        with patch.object(agent, 'MarketConfig', side_effect=RuntimeError('configuration failed')):
            code, current = self.fixture.run_agent()
        self.assertEqual(code, 1)
        attempt = current.pop('last_attempt')
        self.assertEqual(current, self.healthy)
        self.assertEqual(attempt['session'], self.healthy['as_of'])
        self.assertTrue(attempt['errors'])

    def test_older_failure_cannot_replace_newer_valid_report(self):
        with patch.object(sessions, 'utc_now', return_value=pd.Timestamp('2026-09-24T21:00Z')), redirect_stdout(io.StringIO()):
            agent._publish_failure(['older retry failed'], session='2026-09-23', allow_retain=True)
        current = json.loads(agent.READ_FILE.read_text())
        self.assertEqual(current.pop('last_attempt')['session'], '2026-09-23')
        self.assertEqual(current, self.healthy)

    def test_json_non_object_previous_state_is_treated_as_unavailable(self):
        agent.READ_FILE.write_text('[]')
        code, current = self.fixture.run_agent({s:f for s,f in self.fixture.bars.items() if s != 'SPY'})
        self.assertEqual(code, 1)
        self.assertFalse(current['healthy'])


if __name__ == '__main__':
    unittest.main()
