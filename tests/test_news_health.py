"""Offline checks for validated, symbol-tagged Yahoo samples and diagnostics."""
import argparse
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

import agent
from mbot import events, news, report, sessions
from mbot.config import INDEXES, RATES_AND_FEAR, SECTORS, MarketConfig


CLOSE = pd.Timestamp('2026-09-24T20:00:00Z')


def article(title='Example report', when='2026-09-24T19:00:00Z'):
    return {'relatedTickers': ['AAA', 'BBB'], 'content': {'title': title, 'pubDate': when,
                        'provider': {'displayName': 'Example Wire'},
                        'canonicalUrl': {'url': 'https://example.test/story'}}}


class NewsHealthTests(unittest.TestCase):
    def provider_response(self, status, payload):
        return SimpleNamespace(status_code=status, text='JSON response',
                               json=MagicMock(return_value=payload))

    def actual_provider_fetch(self, response, *, opt_in=True):
        """Exercise the real search adapter without contacting Yahoo."""
        diagnostics = {}
        with patch('curl_cffi.requests.get', return_value=response) as get:
            kwargs = {'diagnostics': diagnostics} if opt_in else {}
            result = news.fetch_headlines(['AAA'], end=CLOSE, pause=0,
                                          detailed=True, **kwargs)
        self.assertEqual(get.call_count, 1)
        self.assertEqual(get.call_args.kwargs['timeout'], news.REQUEST_TIMEOUT)
        self.assertEqual(get.call_args.kwargs['params']['q'], 'AAA')
        self.assertEqual(response.json.call_count, 1 if response.status_code == 200 else 0)
        return result, diagnostics

    def test_http_404_is_unavailable_even_without_diagnostics(self):
        result, diagnostics = self.actual_provider_fetch(self.provider_response(
            404, {'message': 'provider endpoint is unavailable'}))
        legacy, _ = self.actual_provider_fetch(self.provider_response(
            404, {'message': 'provider endpoint is unavailable'}), opt_in=False)
        self.assertEqual(result, legacy)
        self.assertEqual(result, ({}, {}, ['AAA'], []))
        self.assertFalse(diagnostics['symbols']['AAA']['observed_zero_in_window'])
        self.assertEqual(diagnostics['symbols']['AAA']['provider_response']['http_status'], 404)
        self.assertEqual(diagnostics['provider_http_error_symbol_count'], 1)
        self.assertNotIn('endpoint is unavailable', json.dumps(diagnostics))

    def test_successful_response_is_observed_without_extra_request_or_decode(self):
        result, diagnostics = self.actual_provider_fetch(self.provider_response(
            200, {'news': [article()]}))
        self.assertEqual(result[0], {'AAA': 1})
        self.assertTrue(diagnostics['symbols']['AAA']['provider_response']['payload_valid'])
        self.assertEqual(diagnostics['provider_http_error_symbol_count'], 0)

    def test_true_http200_empty_and_out_of_window_samples_remain_valid_zero(self):
        for items in ([], [article('Older', '2026-09-23T19:00:00Z')]):
            with self.subTest(items=items):
                result, diagnostics = self.actual_provider_fetch(self.provider_response(200, {'news':items}))
                self.assertEqual(result, ({'AAA':0}, {'AAA':[]}, [], []))
                self.assertTrue(diagnostics['symbols']['AAA']['provider_response']['payload_valid'])

    def test_failed_or_malformed_provider_responses_never_become_valid_zero(self):
        for status, payload in ((429, {'message':'rate limited'}), (500, {'message':'error'}),
                                (200, {}), (200, {'news':None}), (200, {'news':[], 'error':'failed'})):
            with self.subTest(status=status, payload=payload):
                result, diagnostics = self.actual_provider_fetch(self.provider_response(status, payload))
                self.assertEqual(result, ({}, {}, ['AAA'], []))
                self.assertFalse(diagnostics['symbols']['AAA']['observed_zero_in_window'])

    def test_search_results_need_explicit_symbol_relevance(self):
        unrelated = article()
        unrelated['relatedTickers'] = ['OTHER']
        result, _ = self.fetch({'AAA':[unrelated]})
        self.assertEqual(result, ({'AAA':0}, {'AAA':[]}, [], []))
        missing = article()
        missing.pop('relatedTickers')
        result, _ = self.fetch({'AAA':[missing]})
        self.assertEqual(result, ({}, {}, ['AAA'], []))

    def test_malformed_titles_or_missing_publisher_are_unavailable(self):
        for title, publisher in (({'instructions':'not a title'}, 'Wire'), ('  ', 'Wire'), ('Valid title', '')):
            item = article()
            item['content']['title'] = title
            item['content']['provider']['displayName'] = publisher
            with self.subTest(title=title, publisher=publisher):
                result, _ = self.fetch({'AAA':[item]})
                self.assertEqual(result, ({}, {}, ['AAA'], []))

    def test_capped_unordered_sample_never_claims_complete_window(self):
        items = [article('Old', '2026-09-23T19:00:00Z')] + [article()] * (news.FETCH_LIMIT-1)
        result, diagnostics = self.fetch({'AAA':items})
        self.assertEqual(result[0], {})
        self.assertEqual(result[3], ['AAA'])
        self.assertFalse(diagnostics['symbols']['AAA']['returned_newest_first'])

    def test_exhausted_batch_budget_marks_unqueried_symbols_unavailable(self):
        elapsed = [0]
        def fetch(symbol, observation):
            elapsed[0] = news.FETCH_BUDGET + 1
            return [article()]
        diagnostics = {}
        with patch.object(news.time, 'monotonic', side_effect=lambda:elapsed[0]), \
                patch.object(news, '_search_news', side_effect=fetch) as get:
            counts, _, failed, _ = news.fetch_headlines(['AAA','BBB'], end=CLOSE, pause=0, diagnostics=diagnostics)
        get.assert_called_once()
        self.assertEqual(counts, {'AAA':1})
        self.assertEqual(failed, ['BBB'])
        self.assertFalse(diagnostics['symbols']['BBB']['observed_zero_in_window'])
        self.assertEqual(diagnostics['symbols']['BBB']['provider_response']['request_skipped'], 'batch_budget')

    def test_unreadable_http_status_cannot_validate_a_sample(self):
        class Response:
            text = 'JSON response'
            json = MagicMock(return_value={'news': [article()]})

            @property
            def status_code(self):
                raise RuntimeError('diagnostic read failed with private-value')

        diagnostics = {}
        with patch('curl_cffi.requests.get', return_value=Response()):
            result = news.fetch_headlines(['AAA'], end=CLOSE, pause=0, diagnostics=diagnostics)
        self.assertEqual(result[0], {})
        self.assertEqual(result[2], ['AAA'])
        self.assertIsNone(diagnostics['symbols']['AAA']['provider_response'])
        self.assertNotIn('private-value', json.dumps(diagnostics))

    def test_http_failure_warns_and_keeps_scheduled_repair_eligible(self):
        def observe(symbol, observation):
            observation.update(http_status=404, http_success=False)
            raise ValueError('provider HTTP failure')

        with patch.object(news, '_search_news', side_effect=observe):
            saved, _, _, stdout, loader, rows, prior = self.run_read(lambda: [], scheduled_runs=[1], mock_provider=False)
        self.assertTrue(saved['healthy'])
        self.assertEqual(saved['status'], 'complete_with_warnings')
        self.assertEqual(saved['warnings'], ['Headlines unavailable for 2 symbols'])
        self.assertEqual(saved['news']['diagnostics']['provider_http_error_symbol_count'], 2)
        self.assertEqual(len(saved['news']['data_notes']), 2)
        self.assertNotIn('retry skipped', stdout)
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(rows, [prior])
        self.assertEqual(saved['stock_research']['coverage']['headline_symbols'], [])

    def fetch(self, responses, *, opt_in=True):
        def fetch(symbol, observation):
            value = responses[symbol]
            if isinstance(value, Exception):
                raise value
            return value
        diagnostics = {}
        kwargs = {'diagnostics': diagnostics} if opt_in else {}
        with patch.object(news, '_search_news', side_effect=fetch):
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
        result, diagnostics = self.fetch({'AAA': [{'relatedTickers':['AAA'], 'content': {'title': 'Missing date'}}]})
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

    def run_read(self, get_news, *, scheduled_runs=(), recovered=(), mock_provider=True):
        """Run one offline read, then optional scheduled repeats, in a temp state dir."""
        cfg = MarketConfig(watchlist=['AAA', 'BBB'])
        dates = sessions.schedule('2025-01-01', '2026-09-24').index
        close = pd.Series(range(len(dates)), index=dates).astype(float) + 100
        frame = pd.DataFrame({'open': close, 'high': close + 1, 'low': close - 1,
                              'close': close, 'volume': 1000}, index=dates)
        symbols = set(cfg.watchlist) | set(INDEXES) | set(SECTORS) | set(RATES_AND_FEAR)
        bars = {symbol: frame.copy() for symbol in symbols}
        bars['^VIX'].loc[:, ['open', 'high', 'low', 'close']] = [14, 15, 13, 14]
        for symbol in recovered:
            bars[symbol].attrs['price_recovery'] = {'as_of': '2026-09-24', 'source': 'intraday'}
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
            loader = stack.enter_context(patch.object(agent.datamod, 'load_session', return_value=bars))
            if mock_provider:
                stack.enter_context(patch.object(news, '_search_news', side_effect=lambda symbol, observation: get_news()))
            stack.enter_context(patch.object(news.time, 'sleep'))
            stdout = stack.enter_context(redirect_stdout(io.StringIO()))
            args = argparse.Namespace(cached=False, no_news=False, scheduled=False)
            self.assertEqual(agent.cmd_run(args), 0)
            for _ in scheduled_runs:
                self.assertEqual(agent.cmd_run(argparse.Namespace(
                    cached=False, no_news=False, scheduled=True)), 0)
            saved = json.loads(agent.READ_FILE.read_text())
            published = json.loads((report.SITE / 'market.json').read_text())
            page = (report.SITE / 'index.html').read_text()
            rows = [json.loads(line) for line in agent.NEWS_FILE.read_text().splitlines()]
            return saved, published, page, stdout.getvalue(), loader, rows, prior

    def test_report_persists_data_note_and_diagnostics_without_changing_risk(self):
        saved, published, page, stdout, _, rows, prior = self.run_read(lambda: [])
        self.assertEqual(saved, published)
        self.assertTrue(saved['healthy'])
        self.assertEqual(saved['risk']['word'], 'risk-on')
        self.assertEqual(saved['risk']['score'], 4)
        # The universe-zero observation is a report-only data note, not a report warning.
        self.assertEqual(saved['status'], 'complete')
        self.assertEqual(saved['warnings'], [])
        self.assertEqual(len(saved['news']['data_notes']), 1)
        self.assertIn('coverage is unconfirmed', saved['news']['data_notes'][0])
        self.assertTrue(saved['news']['diagnostics']['universe_zero_in_window_observed'])
        self.assertEqual(saved['news']['diagnostics']['symbols']['AAA']['status'], 'empty_response')
        self.assertIn('DATA NOTE: No in-window headlines observed', stdout)
        self.assertIn('Headline feed notes', page)
        self.assertEqual(rows[0], prior)
        self.assertEqual(rows[-1]['counts'], {'AAA': 0, 'BBB': 0})

    def test_universe_zero_note_does_not_block_scheduled_retry_skip(self):
        saved, _, _, stdout, loader, _, _ = self.run_read(lambda: [], scheduled_runs=[1])
        self.assertIn('retry skipped', stdout)
        self.assertEqual(loader.call_count, 1)
        self.assertTrue(saved['news']['data_notes'])

    def test_report_warning_still_triggers_scheduled_retry(self):
        def failing():
            raise RuntimeError('provider down')
        saved, _, _, stdout, loader, _, _ = self.run_read(failing, scheduled_runs=[1])
        self.assertNotIn('retry skipped', stdout)
        self.assertEqual(loader.call_count, 2)
        self.assertIn('Headlines unavailable for 2 symbols', saved['warnings'])

    def test_batch_metrics_zero_universe(self):
        saved, *_ = self.run_read(lambda: [])
        health = saved['news_health']
        self.assertEqual(health['status'], 'zero_universe')
        self.assertEqual((health['session_article_count'], health['symbols_with_articles']), (0, 0))
        self.assertTrue(health['universe_zero_in_window_observed'])
        metrics = saved['batch_metrics']
        self.assertEqual(metrics['headline_symbols_requested'], 2)
        self.assertEqual(metrics['price_bars']['watchlist'], {'intraday_rebuilt': 0, 'daily': 2})
        self.assertEqual(metrics['price_bars']['universe']['intraday_rebuilt'], 0)
        self.assertEqual(metrics['price_bars']['universe']['daily'], saved['universe']['loaded'])

    def test_batch_metrics_ok_and_intraday_counts_do_not_change_research_fields(self):
        saved, *_ = self.run_read(lambda: [article()], recovered=['AAA', 'SPY'])
        self.assertEqual(saved['news_health']['status'], 'ok')
        self.assertEqual(saved['news_health']['session_article_count'], 2)
        self.assertEqual(saved['news_health']['symbols_with_articles'], 2)
        self.assertEqual(saved['batch_metrics']['price_bars']['watchlist'],
                         {'intraday_rebuilt': 1, 'daily': 1})
        self.assertEqual(saved['batch_metrics']['price_bars']['universe']['intraday_rebuilt'], 2)
        self.assertTrue(saved['healthy'] and saved['finalized'])
        self.assertEqual(saved['status'], 'complete_with_warnings')  # price recovery warning only
        self.assertEqual(len(saved['warnings']), 1)
        for row in saved['stock_research']['findings']:
            self.assertEqual(row['headline_status'], 'available')

    def test_batch_metrics_partial_on_provider_failure(self):
        def failing():
            raise RuntimeError('provider down')
        saved, *_ = self.run_read(failing)
        self.assertEqual(saved['news_health']['status'], 'partial')
        self.assertEqual(saved['news_health']['failed_symbols'], 2)
        self.assertEqual(saved['batch_metrics']['session_article_count'], 0)

if __name__ == '__main__':
    unittest.main()
