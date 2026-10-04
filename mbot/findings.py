"""Versioned, descriptive stock evidence for the trader's own screening layer."""
from copy import deepcopy

from . import news, sessions

SCHEMA_VERSION = 1


def build(*, as_of, symbols, bars, counts, headlines, baseline, events, sectors,
          failed=(), capped=(), skipped=False, cfg):
    requested = sorted(set(symbols))
    # Current close alone cannot support an ATR event when its recent history
    # has gaps. Make those gaps visible instead of calling a multi-day move daily.
    import pandas as pd
    end = pd.Timestamp(as_of)
    recent = sessions.schedule(str((end - pd.Timedelta(days=60)).date()), as_of).index[-(cfg.atr_period + 2):]
    price_symbols = [s for s in requested if s in bars and all(d in bars[s].index for d in recent)]
    recovered = {s: deepcopy(bars[s].attrs['price_recovery']) for s in price_symbols
                 if bars[s].attrs.get('price_recovery', {}).get('as_of') == as_of}
    headline_symbols = [s for s in requested if s in counts and not skipped]
    rows = []
    for sym in requested:
        warnings = []
        price_ok = sym in price_symbols
        headline_status = ('skipped' if skipped else 'capped' if sym in capped else
                           'available' if sym in headline_symbols else 'unavailable')
        if not price_ok:
            warnings.append('Current price or consecutive recent bars unavailable; unusual moves not assessed.')
        if sym in recovered:
            warnings.append('Latest prices recovered from complete regular-session 30-minute bars; '
                            'prices and volume may differ from finalized daily prints.')
        if headline_status != 'available':
            warnings.append(f'Headline coverage {headline_status}; missing headlines are not evidence of no news.')
        baseline_status = ('unavailable' if headline_status != 'available' else
                           'ready' if sym in baseline else 'warming_up')
        if baseline_status == 'warming_up':
            warnings.append('Fewer than five valid prior headline observations; headline bursts not assessed.')
        row_events = []
        for event in events:
            if event['symbol'] != sym or event['date'] != as_of or not price_ok:
                continue
            row_events.append({k: event[k] for k in ('kind', 'move_atr', 'move_pct', 'gap_pct',
                               'headlines_today', 'headlines_usual', 'news_assessment')})
        rows.append({
            'id': f'{as_of}:{sym}', 'symbol': sym, 'as_of': as_of,
            'price_status': 'available' if price_ok else 'unavailable',
            'headline_status': headline_status, 'news_baseline_status': baseline_status,
            'headlines': deepcopy(headlines.get(sym, []))[:20], 'events': row_events,
            # We do not have a verified company-to-sector mapping. These are
            # dated market comparisons, never a claim about this stock's sector.
            'sector_context': {'scope': 'market', 'as_of': as_of, 'comparisons': deepcopy(sectors)},
            'warnings': warnings,
        })
        if sym in recovered:
            rows[-1]['price_recovery'] = deepcopy(recovered[sym])
    missing_price = sorted(set(requested) - set(price_symbols))
    missing_news = sorted(set(requested) - set(headline_symbols))
    warnings = []
    if missing_price:
        warnings.append(f'Unusual moves unassessed for {len(missing_price)} symbols.')
    if missing_news:
        warnings.append(f'Incomplete headline coverage for {len(missing_news)} symbols.')
    if len(sectors) != 11:
        warnings.append('Some sector comparisons are unavailable.')
    if recovered:
        warnings.append(f'Latest prices reconstructed from intraday bars for {len(recovered)} symbols.')
    result = {
        'schema_version': SCHEMA_VERSION, 'as_of': as_of,
        'source': {'provider': 'Yahoo Finance', 'price_method': 'adjusted_daily_ohlcv',
                   'news_method': news.METHOD},
        'coverage': {'requested_symbols': requested, 'price_symbols': price_symbols,
                     'headline_symbols': headline_symbols, 'missing_price_symbols': missing_price,
                     'missing_headline_symbols': missing_news,
                     'capped_headline_symbols': sorted(set(capped) & set(requested))},
        'warnings': warnings, 'findings': rows,
    }
    if recovered:
        result['source']['price_method'] = 'adjusted_daily_ohlcv_with_complete_intraday_recovery'
        result['source']['price_recovery'] = recovered
    return result
