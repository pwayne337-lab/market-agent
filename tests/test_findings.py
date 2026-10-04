"""Offline checks for dated, source-attributed stock research publication."""
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

import agent
from mbot import events, findings, news, report, sessions
from mbot.config import INDEXES, RATES_AND_FEAR, SECTORS, MarketConfig


AS_OF = '2026-09-24'
CLOSE = pd.Timestamp('2026-09-24T20:00:00Z')


def bars():
    dates = sessions.schedule('2025-01-01', AS_OF).index
    return pd.DataFrame(dict(open=100., high=101., low=99., close=100., volume=1000.), index=dates)


def article(ident='one', when='2026-09-24T19:00:00Z', url='https://example.test/story'):
    return {'id': ident, 'content': {'title': f'Story {ident}', 'pubDate': when,
            'provider': {'displayName': 'Example Wire'}, 'canonicalUrl': {'url': url}}}


def event(symbol='AAA', day=AS_OF):
    return {'symbol': symbol, 'date': day, 'kind': 'move', 'move_atr': 3.5,
            'move_pct': 8.0, 'gap_pct': 1.5, 'headlines_today': 1,
            'headlines_usual': 1.0, 'news_assessment': 'assessed'}


class FindingsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = MarketConfig(watchlist=['AAA', 'BBB', 'CCC'])
        self.frame = bars()
        self.sectors = [{'symbol': symbol, 'name': name, 'rank': rank,
                         'rs_5d': 0.5, 'rs_21d': 1.0, 'ret_21d': 2.0}
                        for rank, (symbol, name) in enumerate(SECTORS.items(), 1)]
        self.headline = {'title': 'Company announces product',
                         'published_at': '2026-09-24T19:00:00+00:00',
                         'publisher': 'Example Wire', 'url': 'https://example.test/story',
                         'source': 'Yahoo Finance'}

    def build(self, **changes):
        kwargs = dict(as_of=AS_OF, symbols=self.cfg.watchlist,
                      bars={s: self.frame for s in self.cfg.watchlist},
                      counts={s: 0 for s in self.cfg.watchlist}, headlines={},
                      baseline={s: 1 for s in self.cfg.watchlist}, events=[],
                      sectors=self.sectors, cfg=self.cfg)
        kwargs.update(changes)
        return findings.build(**kwargs)

    def test_each_symbol_has_exact_date_identity_source_and_coverage(self):
        result = self.build(symbols=['BBB', 'AAA', 'AAA'],
                            headlines={'AAA': [self.headline]}, events=[event()])
        self.assertEqual(result['schema_version'], 1)
        self.assertEqual(result['as_of'], AS_OF)
        self.assertEqual(result['source'], {'provider': 'Yahoo Finance',
                         'price_method': 'adjusted_daily_ohlcv', 'news_method': news.METHOD})
        self.assertEqual(result['coverage']['requested_symbols'], ['AAA', 'BBB'])
        self.assertEqual(result['coverage']['price_symbols'], ['AAA', 'BBB'])
        self.assertEqual(result['coverage']['headline_symbols'], ['AAA', 'BBB'])
        self.assertEqual(result['coverage']['missing_price_symbols'], [])
        self.assertEqual(result['coverage']['missing_headline_symbols'], [])
        self.assertEqual([r['id'] for r in result['findings']], [AS_OF + ':AAA', AS_OF + ':BBB'])
        row = result['findings'][0]
        self.assertEqual(row['symbol'], 'AAA')
        self.assertEqual(row['as_of'], AS_OF)
        self.assertEqual(row['headlines'], [self.headline])
        self.assertEqual(row['events'][0]['move_atr'], 3.5)
        self.assertEqual(row['news_baseline_status'], 'ready')
        self.assertEqual(result['warnings'], [])
        self.assertNotIn('earnings', row)

    def test_events_are_exact_symbol_and_session_only(self):
        result = self.build(events=[event(), event('BBB'), event(day='2026-09-23'), event('OUTSIDE')])
        rows = {r['symbol']: r for r in result['findings']}
        self.assertEqual(len(rows['AAA']['events']), 1)
        self.assertEqual(len(rows['BBB']['events']), 1)
        self.assertEqual(rows['CCC']['events'], [])
        self.assertNotIn('OUTSIDE', rows)

    def test_recovered_prices_keep_symbol_specific_provenance_and_warning(self):
        recovered = self.frame.copy()
        metadata = {'as_of': AS_OF, 'method': 'complete_regular_session_30m',
                    'interval': '30m', 'anchor_session': '2026-09-23'}
        recovered.attrs['price_recovery'] = metadata
        result = self.build(bars={'AAA': recovered, 'BBB': self.frame}, events=[event()])
        rows = {r['symbol']: r for r in result['findings']}
        self.assertEqual(result['source']['price_recovery'], {'AAA': metadata})
        self.assertEqual(rows['AAA']['price_recovery'], metadata)
        self.assertEqual(rows['AAA']['price_status'], 'available')
        self.assertTrue(any('finalized daily prints' in w for w in rows['AAA']['warnings']))
        self.assertNotIn('price_recovery', rows['BBB'])
        metadata['as_of'] = 'changed'
        self.assertEqual(rows['AAA']['price_recovery']['as_of'], AS_OF)

    def test_stale_recovery_metadata_is_not_attached_to_current_report(self):
        recovered = self.frame.copy()
        recovered.attrs['price_recovery'] = {'as_of': '2026-09-23'}
        result = self.build(bars={'AAA': recovered})
        self.assertNotIn('price_recovery', result['source'])
        self.assertNotIn('price_recovery', result['findings'][0])

    def test_sector_context_is_dated_market_context_without_company_mapping(self):
        result = self.build()
        for row in result['findings']:
            self.assertEqual(row['sector_context'], {
                'scope': 'market', 'as_of': AS_OF, 'comparisons': self.sectors})
            self.assertNotIn('sector', row)
        self.sectors[0]['rs_5d'] = 999
        self.assertEqual(result['findings'][0]['sector_context']['comparisons'][0]['rs_5d'], 0.5)

    def test_missing_sector_comparisons_are_warned(self):
        result = self.build(sectors=self.sectors[:3])
        self.assertIn('Some sector comparisons are unavailable.', result['warnings'])
        self.assertEqual(len(result['findings'][0]['sector_context']['comparisons']), 3)

    def test_incomplete_history_never_produces_price_events(self):
        histories = {'stale': self.frame.iloc[:-1],
                     'missing_previous': self.frame.drop(self.frame.index[-2]),
                     'missing_atr_input': self.frame.drop(self.frame.index[-10]),
                     'too_short': self.frame.iloc[-5:]}
        for name, history in histories.items():
            with self.subTest(name=name):
                result = self.build(bars={'AAA': history}, events=[event()],
                                    headlines={'AAA': [self.headline]})
                row = result['findings'][0]
                self.assertEqual(row['price_status'], 'unavailable')
                self.assertEqual(row['events'], [])
                self.assertEqual(row['headlines'], [self.headline])
                self.assertIn('AAA', result['coverage']['missing_price_symbols'])
                self.assertTrue(any('unusual moves not assessed' in w for w in row['warnings']))

    def test_missing_news_and_zero_news_remain_distinct(self):
        result = self.build(counts={'AAA': 0}, failed=['BBB', 'CCC'])
        rows = {r['symbol']: r for r in result['findings']}
        self.assertEqual(rows['AAA']['headline_status'], 'available')
        self.assertEqual(rows['AAA']['warnings'], [])
        self.assertEqual(rows['BBB']['headline_status'], 'unavailable')
        self.assertEqual(rows['BBB']['news_baseline_status'], 'unavailable')
        self.assertTrue(any('not evidence of no news' in w for w in rows['BBB']['warnings']))
        self.assertEqual(result['coverage']['headline_symbols'], ['AAA'])
        self.assertEqual(result['coverage']['missing_headline_symbols'], ['BBB', 'CCC'])

    def test_skipped_news_stays_explicit_and_unavailable(self):
        result = self.build(counts={}, skipped=True)
        self.assertEqual(result['coverage']['headline_symbols'], [])
        for row in result['findings']:
            self.assertEqual(row['headline_status'], 'skipped')
            self.assertEqual(row['news_baseline_status'], 'unavailable')
            self.assertEqual(row['headlines'], [])

    def test_warming_baseline_and_headline_limit(self):
        headlines = [dict(self.headline, title=f'Story {i}') for i in range(30)]
        result = self.build(baseline={}, headlines={'AAA': headlines})
        row = result['findings'][0]
        self.assertEqual(row['news_baseline_status'], 'warming_up')
        self.assertTrue(any('Fewer than five' in w for w in row['warnings']))
        self.assertEqual(len(row['headlines']), 20)
        headlines[0]['title'] = 'Changed later'
        self.assertEqual(row['headlines'][0]['title'], 'Story 0')

    def fetch(self, items):
        ticker = MagicMock()
        ticker.get_news.return_value = items
        with patch('yfinance.Ticker', return_value=ticker):
            result = news.fetch_headlines(['AAA'], end=CLOSE, pause=0, detailed=True)
        ticker.get_news.assert_called_once_with(count=news.FETCH_LIMIT)
        return result

    def test_headlines_have_dated_sources_links_and_session_window(self):
        items = [article(), article(), article('late', '2026-09-24T20:01:00Z'),
                 article('old', '2026-09-23T20:00:00Z'), article('at_close', '2026-09-24T20:00:00Z')]
        counts, headlines, failed, capped = self.fetch(items)
        self.assertEqual(counts, {'AAA': 2})
        self.assertEqual(failed, [])
        self.assertEqual(capped, [])
        self.assertEqual(headlines['AAA'][0], dict(self.headline, title='Story one'))
        for row in headlines['AAA']:
            self.assertGreater(pd.Timestamp(row['published_at']), CLOSE - pd.Timedelta(days=1))
            self.assertLessEqual(pd.Timestamp(row['published_at']), CLOSE)

    def test_capped_sample_retains_headlines_but_never_claims_complete_count(self):
        items = [article(str(i)) for i in range(news.FETCH_LIMIT)]
        counts, headlines, failed, capped = self.fetch(items)
        self.assertEqual(counts, {})
        self.assertEqual(failed, [])
        self.assertEqual(capped, ['AAA'])
        self.assertEqual(len(headlines['AAA']), 20)
        result = self.build(counts=counts, headlines=headlines, capped=capped)
        row = result['findings'][0]
        self.assertEqual(row['headline_status'], 'capped')
        self.assertEqual(row['news_baseline_status'], 'unavailable')
        self.assertEqual(len(row['headlines']), 20)
        self.assertNotIn('AAA', result['coverage']['headline_symbols'])
        self.assertIn('AAA', result['coverage']['missing_headline_symbols'])
        self.assertEqual(result['coverage']['capped_headline_symbols'], ['AAA'])

    def test_malformed_news_is_unavailable_and_unsafe_url_is_removed(self):
        for malformed in ({'unexpected': 'schema'}, [{'content': {'title': 'Undated'}}],
                          [article(when='2026-09-24T19:00:00')]):
            with self.subTest(malformed=malformed):
                counts, headlines, failed, capped = self.fetch(malformed)
                self.assertEqual(counts, {})
                self.assertEqual(headlines, {})
                self.assertEqual(failed, ['AAA'])
                self.assertEqual(capped, [])
        counts, headlines, _, _ = self.fetch([article(url='javascript:alert(1)')])
        self.assertEqual(counts, {'AAA': 1})
        self.assertIsNone(headlines['AAA'][0]['url'])


class FindingsPublicationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        for obj, name, value in [(agent, 'STATE_DIR', root / 'state'),
                                 (agent, 'READ_FILE', root / 'state/market.json'),
                                 (agent, 'NEWS_FILE', root / 'state/news_counts.jsonl'),
                                 (events, 'STATE_DIR', root / 'state'),
                                 (events, 'EVENTS_FILE', root / 'state/events.jsonl'),
                                 (report, 'SITE', root / 'site')]:
            mocked = patch.object(obj, name, value)
            mocked.start()
            self.addCleanup(mocked.stop)
        self.cfg = MarketConfig(watchlist=['AAA', 'BBB', 'CCC'])
        self.bars = {s: bars() for s in set(self.cfg.watchlist) | set(INDEXES) | set(SECTORS) | set(RATES_AND_FEAR)}
        self.bars['^VIX']['close'] = 20.
        self.bars['AAA'].loc[pd.Timestamp(AS_OF), ['high', 'close']] = [111., 110.]
        self.args = argparse.Namespace(cached=False, no_news=False, scheduled=False)

    def run_agent(self, news_result=None):
        if news_result is None:
            news_result = ({s: 0 for s in self.cfg.watchlist}, {}, [], [])
        with patch.object(agent, 'MarketConfig', return_value=self.cfg), \
             patch.object(sessions, 'utc_now', return_value=pd.Timestamp('2026-09-24T20:47Z')), \
             patch.object(agent.datamod, 'load_session', return_value=self.bars), \
             patch.object(news, 'fetch_headlines', return_value=news_result) as fetch, \
             contextlib.redirect_stdout(io.StringIO()):
            code = agent.cmd_run(self.args)
        saved = json.loads(agent.READ_FILE.read_text())
        self.assertEqual(saved, json.loads((report.SITE / 'market.json').read_text()))
        return code, saved, fetch

    def test_publication_includes_events_and_source_attributed_findings(self):
        headline = {'title': 'Example company story', 'published_at': '2026-09-24T19:00:00+00:00',
                    'publisher': 'Example Wire', 'url': 'https://example.test/story', 'source': 'Yahoo Finance'}
        code, read, fetch = self.run_agent(({'AAA': 1, 'BBB': 0, 'CCC': 0}, {'AAA': [headline]}, [], []))
        self.assertEqual(code, 0)
        self.assertTrue(read['healthy'])
        self.assertTrue(read['finalized'])
        self.assertEqual(read['as_of'], AS_OF)
        self.assertGreater(pd.Timestamp(read['expires_at']), CLOSE)
        fetch.assert_called_once_with(self.cfg.watchlist, end=CLOSE, detailed=True)
        finding = read['stock_research']['findings'][0]
        self.assertEqual(finding['headlines'], [headline])
        self.assertEqual(len(finding['events']), 1)
        self.assertEqual(finding['events'][0]['kind'], 'move')
        self.assertEqual(read['events_today'][0]['top_headlines'], [headline['title']])

    def test_partial_headline_report_is_usable_and_explicit(self):
        code, read, _ = self.run_agent(({'AAA': 0}, {}, ['BBB', 'CCC'], []))
        self.assertEqual(code, 0)
        self.assertEqual(read['status'], 'complete_with_warnings')
        self.assertTrue(read['healthy'])
        rows = {r['symbol']: r for r in read['stock_research']['findings']}
        self.assertEqual(rows['AAA']['headline_status'], 'available')
        self.assertEqual(rows['BBB']['headline_status'], 'unavailable')
        self.assertEqual(read['stock_research']['coverage']['missing_headline_symbols'], ['BBB', 'CCC'])

    def test_skipped_headlines_do_not_call_provider(self):
        self.args.cached = True
        code, read, fetch = self.run_agent()
        self.assertEqual(code, 0)
        fetch.assert_not_called()
        self.assertTrue(all(r['headline_status'] == 'skipped' for r in read['stock_research']['findings']))
        self.assertEqual(read['stock_research']['coverage']['headline_symbols'], [])

    def test_failed_read_keeps_previous_report_separate_from_current_research(self):
        _, previous, _ = self.run_agent()
        del self.bars['SPY']
        code, failed, _ = self.run_agent()
        self.assertEqual(code, 1)
        self.assertFalse(failed['healthy'])
        self.assertFalse(failed['finalized'])
        self.assertNotIn('stock_research', failed)
        self.assertEqual(failed['last_good']['stock_research'], previous['stock_research'])


if __name__ == '__main__':
    unittest.main()
