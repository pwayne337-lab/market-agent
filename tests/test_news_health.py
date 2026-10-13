"""Offline checks for additive feed diagnostics without changing research rules."""
import argparse
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

import agent
from mbot import events, news, report, sessions
from mbot.config import INDEXES, RATES_AND_FEAR, SECTORS, MarketConfig


CLOSE = pd.Timestamp('2026-09-24T20:00:00Z')


def article(title='Example report', when='2026-09-24T19:00:00Z'):
    return {'content': {'title': title, 'pubDate': when,
                        'provider': {'displayName': 'Example Wire'},
                        'canonicalUrl': {'url': 'https://example.test/story'}}}


class NewsHealthTests(unittest.TestCase):
    def fetch(self, responses, *, opt_in=True):
        def ticker(symbol):
            result = MagicMock()
            value = responses[symbol]
            if isinstance(value, Exception):
                result.get_news.side_effect = value
            else:
                result.get_news.return_value = value
            return result
        diagnostics = {}
        kwargs = {'diagnostics': diagnostics} if opt_in else {}
        with patch('yfinance.Ticker', side_effect=ticker):
            result = news.fetch_headlines(list(responses), end=CLOSE, pause=0,
                                          detailed=True, **kwargs)
        self.assertEqual(len(result), 4)
        return result, diagnostics

    def test_positive_sample_records_bounds_and_preserves_four_value_api(self):
        inside = article()
        responses = {'AAA': [inside, inside,
                             article('Before window', '2026-09-23T19:00:00Z'),
                             article('After close', '2026-09-24T21:00:00Z')]}
        result, diagnostics = self.fetch(responses)
        without_diagnostics, _ = self.fetch(responses, opt_in=False)
        self.assertEqual(result, without_diagnostics)
        counts, headlines, failed, capped = result
        self.assertEqual(counts, {'AAA': 1})
        self.assertEqual(len(headlines['AAA']), 1)
        self.assertEqual((failed, capped), ([], []))
        row = diagnostics['symbols']['AAA']
        self.assertEqual(row['returned_item_count'], 4)
        self.assertEqual(row['matched_in_window_count'], 1)
        self.assertEqual(row['oldest_returned_at'], '2026-09-23T19:00:00+00:00')
        self.assertEqual(row['newest_returned_at'], '2026-09-24T21:00:00+00:00')
        self.assertTrue(row['sample_reaches_window_start'])
        self.assertTrue(row['parse_complete'])
        self.assertFalse(row['published_news_coverage_confirmed'])
        self.assertEqual(diagnostics['warnings'], [])

    def test_quiet_single_symbol_stays_valid_without_universe_warning(self):
        result, diagnostics = self.fetch({'AAA': [article('Older', '2026-09-23T19:00:00Z')]})
        self.assertEqual(result[0], {'AAA': 0})
        row = diagnostics['symbols']['AAA']
        self.assertEqual(row['status'], 'no_items_in_window')
        self.assertTrue(row['observed_zero_in_window'])
        self.assertTrue(row['sample_reaches_window_start'])
        self.assertFalse(row['published_news_coverage_confirmed'])
        self.assertEqual(diagnostics['warnings'], [])

    def test_empty_universe_is_visible_without_changing_counts(self):
        result, diagnostics = self.fetch({'AAA': [], 'BBB': []})
        self.assertEqual(result, ({'AAA': 0, 'BBB': 0}, {'AAA': [], 'BBB': []}, [], []))
        self.assertTrue(diagnostics['universe_zero_in_window_observed'])
        self.assertFalse(diagnostics['published_news_coverage_confirmed'])
        self.assertEqual(len(diagnostics['warnings']), 1)
        self.assertIn('coverage is unconfirmed', diagnostics['warnings'][0])
        for row in diagnostics['symbols'].values():
            self.assertEqual(row['returned_item_count'], 0)
            self.assertIsNone(row['oldest_returned_at'])
            self.assertEqual(row['status'], 'empty_response')

    def test_out_of_window_universe_differs_from_empty_responses(self):
        result, diagnostics = self.fetch({
            'AAA': [article('Older', '2026-09-23T19:00:00Z')],
            'BBB': [article('Newer', '2026-09-24T21:00:00Z')],
        })
        self.assertEqual(result[0], {'AAA': 0, 'BBB': 0})
        self.assertTrue(diagnostics['universe_zero_in_window_observed'])
        self.assertEqual(len(diagnostics['warnings']), 1)
        self.assertTrue(diagnostics['symbols']['AAA']['sample_reaches_window_start'])
        self.assertFalse(diagnostics['symbols']['BBB']['sample_reaches_window_start'])
        for row in diagnostics['symbols'].values():
            self.assertEqual(row['returned_item_count'], 1)
            self.assertEqual(row['matched_in_window_count'], 0)
            self.assertIsNotNone(row['oldest_returned_at'])
            self.assertEqual(row['status'], 'no_items_in_window')

    def test_malformed_article_records_failure_without_a_valid_zero(self):
        result, diagnostics = self.fetch({'AAA': [{'content': {'title': 'Missing date'}}]})
        self.assertEqual(result, ({}, {}, ['AAA'], []))
        row = diagnostics['symbols']['AAA']
        self.assertEqual(row['returned_item_count'], 1)
        self.assertEqual(row['error_type'], 'ValueError')
        self.assertEqual(row['error_reason'], 'article missing date or title')
        self.assertFalse(row['parse_complete'])
        self.assertFalse(row['observed_zero_in_window'])

    def test_provider_failure_reason_omits_raw_urls_and_credentials(self):
        result, diagnostics = self.fetch({'AAA': RuntimeError(
            'request https://example.test/?token=private-value Bearer secret-value')})
        self.assertEqual(result[2], ['AAA'])
        row = diagnostics['symbols']['AAA']
        self.assertEqual(row['error_type'], 'RuntimeError')
        encoded = json.dumps(diagnostics)
        self.assertNotIn('private-value', encoded)
        self.assertNotIn('secret-value', encoded)
        self.assertNotIn('https://example.test', encoded)
        self.assertIn('raw details omitted', row['error_reason'])

    def test_capped_sample_keeps_existing_count_exclusion(self):
        result, diagnostics = self.fetch({'AAA': [article()] * news.FETCH_LIMIT})
        self.assertEqual(result[0], {})
        self.assertEqual(result[3], ['AAA'])
        row = diagnostics['symbols']['AAA']
        self.assertEqual(row['status'], 'capped')
        self.assertEqual(row['returned_item_count'], news.FETCH_LIMIT)
        self.assertEqual(row['matched_in_window_count'], 1)
        self.assertFalse(row['sample_reaches_window_start'])

    def test_report_persists_warning_and_diagnostics_without_changing_risk(self):
        cfg = MarketConfig(watchlist=['AAA', 'BBB'])
        dates = sessions.schedule('2025-01-01', '2026-09-24').index
        close = pd.Series(range(len(dates)), index=dates).astype(float) + 100
        frame = pd.DataFrame({'open': close, 'high': close + 1, 'low': close - 1,
                              'close': close, 'volume': 1000}, index=dates)
        symbols = set(cfg.watchlist) | set(INDEXES) | set(SECTORS) | set(RATES_AND_FEAR)
        bars = {symbol: frame.copy() for symbol in symbols}
        bars['^VIX'].loc[:, ['open', 'high', 'low', 'close']] = [14, 15, 13, 14]
        args = argparse.Namespace(cached=False, no_news=False, scheduled=False)
        prior = {'date': '2026-09-23', 'method': news.METHOD,
                 'counts': {'AAA': 2, 'BBB': 3}}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            state = root / 'state'
            state.mkdir()
            for module, name, path in [
                (agent, 'STATE_DIR', state), (agent, 'READ_FILE', state / 'market.json'),
                (agent, 'NEWS_FILE', state / 'news_counts.jsonl'),
                (events, 'STATE_DIR', state), (events, 'EVENTS_FILE', state / 'events.jsonl'),
                (report, 'SITE', root / 'site'),
            ]:
                stack.enter_context(patch.object(module, name, path))
            agent.NEWS_FILE.write_text(json.dumps(prior) + '\n')
            stack.enter_context(patch.object(agent, 'MarketConfig', return_value=cfg))
            stack.enter_context(patch.object(sessions, 'utc_now', return_value=CLOSE + pd.Timedelta(minutes=47)))
            stack.enter_context(patch.object(agent.datamod, 'load_session', return_value=bars))
            ticker = MagicMock()
            ticker.get_news.return_value = []
            stack.enter_context(patch('yfinance.Ticker', return_value=ticker))
            stack.enter_context(patch.object(news.time, 'sleep'))
            stdout = stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(agent.cmd_run(args), 0)
            saved = json.loads(agent.READ_FILE.read_text())
            published = json.loads((report.SITE / 'market.json').read_text())
            self.assertEqual(saved, published)
            self.assertTrue(saved['healthy'])
            self.assertEqual(saved['risk']['word'], 'risk-on')
            self.assertEqual(saved['risk']['score'], 4)
            self.assertEqual(saved['status'], 'complete_with_warnings')
            self.assertTrue(saved['news']['diagnostics']['universe_zero_in_window_observed'])
            self.assertEqual(saved['news']['diagnostics']['symbols']['AAA']['status'], 'empty_response')
            self.assertIn('coverage is unconfirmed', stdout.getvalue())
            rows = [json.loads(line) for line in agent.NEWS_FILE.read_text().splitlines()]
            self.assertEqual(rows[0], prior)
            self.assertEqual(rows[-1]['counts'], {'AAA': 0, 'BBB': 0})


if __name__ == '__main__':
    unittest.main()
