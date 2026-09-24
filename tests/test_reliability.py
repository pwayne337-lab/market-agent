import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import agent
from mbot import events, news, regime, report, sessions
from mbot.config import MarketConfig, INDEXES, SECTORS, RATES_AND_FEAR

class Reliability(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for obj, name, value in [(agent,'STATE_DIR',root/'state'),(agent,'READ_FILE',root/'state/market.json'),
            (agent,'NEWS_FILE',root/'state/news_counts.jsonl'),(events,'STATE_DIR',root/'state'),
            (events,'EVENTS_FILE',root/'state/events.jsonl'),(report,'SITE',root/'site')]:
            p = patch.object(obj,name,value);p.start();self.addCleanup(p.stop)
        self.cfg = MarketConfig()
        dates = sessions.schedule('2025-01-01','2026-09-24').index
        x = np.linspace(100,150,len(dates))
        self.frame = pd.DataFrame(dict(open=x,high=x+1,low=x-1,close=x,volume=1000),index=dates)
        symbols = set(self.cfg.watchlist)|set(INDEXES)|set(SECTORS)|set(RATES_AND_FEAR)
        self.bars = {s:self.frame.copy() for s in symbols}
        self.bars['^VIX']['close'] = 20
        self.args = argparse.Namespace(cached=False,no_news=False,scheduled=False)

    def run_agent(self, bars=None, news_result=None):
        result = news_result if news_result is not None else ({s:1 for s in self.cfg.watchlist},{},[],[])
        with patch.object(sessions,'utc_now',return_value=pd.Timestamp('2026-09-24T20:47Z')), \
             patch.object(agent.datamod,'load_universe',return_value=self.bars if bars is None else bars), \
             patch.object(news,'fetch_headlines',return_value=result), contextlib.redirect_stdout(io.StringIO()):
            code = agent.cmd_run(self.args)
        return code,json.loads(agent.READ_FILE.read_text())

    def test_calendar_boundaries(self):
        self.assertIsNone(sessions.today_session('2026-11-26T22:00Z'))
        self.assertFalse(sessions.today_session('2026-09-24T18:57Z')['ready'])
        self.assertTrue(sessions.today_session('2026-11-27T18:15Z')['ready'])
        self.assertEqual(sessions.today_session('2026-11-27T18:15Z')['close'].hour,18)
        self.assertFalse(sessions.today_session('2026-12-01T21:14Z')['ready'])
        self.assertTrue(sessions.today_session('2026-12-01T21:15Z')['ready'])
        self.assertEqual(sessions.follow_dates('2026-11-25',1),'2026-11-27')

    def test_early_run_preserves_research(self):
        with patch.object(sessions,'utc_now',return_value=pd.Timestamp('2026-09-24T18:57Z')),patch.object(agent.datamod,'load_universe') as load:
            self.assertEqual(agent.cmd_run(self.args),0)
            load.assert_not_called()
        self.assertFalse(events.EVENTS_FILE.exists())
        self.assertEqual(json.loads((report.SITE/'market.json').read_text())['status'],'waiting')

    def test_success_and_scheduled_duplicate(self):
        code,read = self.run_agent()
        self.assertEqual(code,0);self.assertTrue(read['finalized']);self.assertTrue(read['healthy'])
        self.assertIn('warming up',read['brief'])
        self.args.scheduled = True
        with patch.object(sessions,'utc_now',return_value=pd.Timestamp('2026-09-24T21:12Z')),patch.object(agent.datamod,'load_universe') as load:
            agent.cmd_run(self.args);load.assert_not_called()

    def test_missing_spy_publishes_current_failure(self):
        self.run_agent()
        bars = {k:v for k,v in self.bars.items() if k!='SPY'}
        code,read=self.run_agent(bars)
        self.assertEqual(code,1);self.assertFalse(read['healthy']);self.assertFalse(read['finalized'])
        self.assertEqual(read['last_good']['as_of'],'2026-09-24')
        self.assertEqual(json.loads((report.SITE/'market.json').read_text()),read)
        self.assertIn('SPY',(report.SITE/'index.html').read_text())

    def test_stale_prices_fail_and_intraday_tail_trimmed(self):
        code,read=self.run_agent({s:d.iloc[:-1] for s,d in self.bars.items()})
        self.assertEqual(code,1);self.assertFalse(read['healthy'])
        future={s:pd.concat([d,pd.DataFrame(dict(open=[999],high=[1000],low=[998],close=[999],volume=[1000]),index=[pd.Timestamp('2026-09-25')])]) for s,d in self.bars.items()}
        code,read=self.run_agent(future)
        self.assertEqual(read['index']['close'],150)

    def test_sector_returns_use_shared_endpoint_dates(self):
        expected = regime.sector_strength(self.bars, self.cfg)
        missing_middle = dict(self.bars, XLK=self.bars['XLK'].drop(self.frame.index[-3]))
        actual = regime.sector_strength(missing_middle, self.cfg)
        self.assertEqual(actual, expected)
        missing_start = dict(self.bars, XLK=self.bars['XLK'].drop(self.frame.index[-22]))
        self.assertNotIn('XLK', [r['symbol'] for r in regime.sector_strength(missing_start, self.cfg)])

    def test_missing_vix_is_unknown(self):
        code,read=self.run_agent({s:d for s,d in self.bars.items() if s!='^VIX'})
        self.assertEqual(code,1);self.assertEqual(read['risk']['word'],'unknown')
        self.assertIsNone(read['risk']['checks'][-1]['ok'])

    def test_missing_news_is_not_zero(self):
        rows=[{'date':str(i),'method':news.METHOD,'counts':{}} for i in range(4)]
        rows += [{'date':'5','method':news.METHOD,'counts':{'S':10}}]
        self.assertNotIn('S',agent._news_baseline(rows,['S'],30))
        rows=[{'date':str(i),'method':news.METHOD,'counts':{'S':10}} for i in range(5)]
        self.assertEqual(agent._news_baseline(rows,['S'],30)['S'],10)
        self.assertEqual(agent._news_baseline([dict(r,method='legacy') for r in rows],['S'],30),{})

    def test_optional_news_gap_is_a_visible_warning(self):
        code,read=self.run_agent(news_result=({}, {}, ['SPY'], []))
        self.assertEqual(code,0)
        self.assertTrue(read['healthy']);self.assertEqual(read['errors'],[])
        self.assertEqual(read['status'],'complete_with_warnings')
        self.assertIn('Headlines unavailable',(report.SITE/'index.html').read_text())

    def test_limited_price_and_news_gaps_do_not_require_user_action(self):
        bars = {s:d for s,d in self.bars.items() if s not in ('GIS','WMB')}
        counts = {s:1 for s in self.cfg.watchlist if s not in ('NVDA','META')}
        code,read=self.run_agent(bars, (counts,{},[],['NVDA','META']))
        self.assertEqual(code,0);self.assertEqual(read['errors'],[])
        self.assertEqual(len(read['warnings']),2)
        self.assertEqual(read['news']['valid_symbols'],191)
        self.assertEqual(read['breadth']['counted'],191)
        self.assertEqual(len(read['sectors']),11)
        self.assertIn('Data notes — no action required',(report.SITE/'index.html').read_text())
        history=agent._load_news_history()
        self.assertNotIn('NVDA',history[-1]['counts'])
        self.args.scheduled = True
        with patch.object(sessions,'utc_now',return_value=pd.Timestamp('2026-09-24T21:12Z')),patch.object(agent.datamod,'load_universe',side_effect=RuntimeError('retry attempted')) as load:
            self.assertEqual(agent.cmd_run(self.args),1)
            load.assert_called_once()

    def test_insufficient_breadth_still_fails(self):
        essential = set(INDEXES)|set(SECTORS)|set(RATES_AND_FEAR)
        code,read=self.run_agent({s:d for s,d in self.bars.items() if s in essential})
        self.assertEqual(code,1);self.assertFalse(read['healthy'])
        self.assertTrue(any('coverage below' in x for x in read['errors']))
        self.assertEqual(read['risk']['word'],'unknown')

    def test_fixed_news_window_dedup_and_cap(self):
        article=lambda ident,dt:{'id':ident,'content':{'title':ident,'pubDate':dt}}
        items=[article('in','2026-09-24T19:00:00Z')]*2+[article('after','2026-09-24T21:00:00Z'),article('old','2026-09-23T19:00:00Z')]
        ticker=MagicMock();ticker.get_news.return_value=items
        with patch('yfinance.Ticker',return_value=ticker):
            counts,titles,failed,capped=news.fetch_headlines(['S'],end=pd.Timestamp('2026-09-24T20:00Z'),pause=0)
        self.assertEqual(counts,{'S':1});self.assertEqual(titles,{'S':['in']})
        ticker.get_news.return_value=[items[0]]*news.FETCH_LIMIT
        with patch('yfinance.Ticker',return_value=ticker):
            counts,_,_,capped=news.fetch_headlines(['S'],end=pd.Timestamp('2026-09-24T20:00Z'),pause=0)
        self.assertEqual(counts,{});self.assertEqual(capped,['S'])
        ticker.get_news.return_value=[items[0]]*99+[items[-1]]
        with patch('yfinance.Ticker',return_value=ticker):
            counts,_,_,capped=news.fetch_headlines(['S'],end=pd.Timestamp('2026-09-24T20:00Z'),pause=0)
        self.assertEqual(counts,{'S':1});self.assertEqual(capped,[])

    def test_missing_target_session_not_shifted(self):
        event={'date':'2026-09-22','symbol':'S','follow':{}}
        frame=self.frame.drop(pd.Timestamp('2026-09-23'))
        events.fill_follow_through([event],{'S':frame},self.cfg)
        self.assertNotIn('1d',event['follow'])

    def test_partial_follow_not_saved(self):
        event={'date':'2026-09-23','symbol':'S','follow':{}}
        with patch.object(sessions,'utc_now',return_value=pd.Timestamp('2026-09-24T18:57Z')):
            events.fill_follow_through([event],{'S':self.frame},self.cfg)
        self.assertEqual(event['follow'],{})

    def test_correlated_results_descriptive_and_versions_separate(self):
        records=[{'date':'2026-09-01','symbol':str(i),'kind':'move','direction':'up','rules_version':2,'follow':{'5d':1}} for i in range(20)]
        rows=events.what_the_record_says(records,self.cfg)
        result=rows[0]['after_5d']
        self.assertEqual(result['distinct_dates'],1)
        self.assertNotIn('real',result);self.assertNotIn('ci',result)
        records.append(dict(records[0],rules_version=1))
        self.assertEqual(len(events.what_the_record_says(records,self.cfg)),2)

    def test_rerun_updates_event_preserving_follow(self):
        old={'date':'2026-09-01','symbol':'S','kind':'move','follow':{'1d':2}}
        new=dict(old,kind='move+news',follow={})
        merged=events.merge([old],[new])
        self.assertEqual(len(merged),1);self.assertEqual(merged[0]['kind'],'move+news')
        self.assertEqual(merged[0]['follow'],{'1d':2})

if __name__=='__main__': unittest.main()
