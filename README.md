# Researcher (the market agent)

Studies the market. Trades nothing.

Every weekday after the close it reads the same 193 names the trading agent
can buy, plus the indexes, the eleven sector ETFs, long bonds and the VIX, and
publishes one page: what the market did today, and what the record says about
what tends to happen after days like it.

It costs nothing to run. Prices and headlines come from Yahoo Finance, the
schedule is GitHub Actions, the page is GitHub Pages. No AI model reads the
news. Where a rule turns numbers into a word, the rule is printed beside it.

## What it publishes

`site/market.json`, and the page built from it:

* **Risk word.** Four yes/no checks: SPY above its 200-day; 50-day above the
  200-day; breadth not weak; VIX not stressed. Three or four is risk-on, zero
  or one risk-off, two mixed. It is a summary of today, never a forecast.
* **Breadth.** Share of the watchlist above its own 200-day and 50-day
  averages; names making 20-day highs and lows. An index up on weak breadth is
  a few large names carrying it.
* **Volatility.** SPY's 20-day realized volatility and its direction; the VIX
  level and its 5-day change.
* **Sectors.** Each sector ETF's return minus SPY's over 5 and 21 sessions,
  ranked. Leadership is what the index is made of.
* **Events.** Every watchlist name that moved at least 2.5 of its own ATRs
  today, or drew three times its usual headline count, or both, with the size
  of the move and the top headlines.

## How it learns

By writing things down and coming back. Each event is stored in
`state/events.jsonl`. One and five sessions later the agent fills in what the
price did next. Once a kind of event (say, "a 3-ATR down day with no news")
has twenty examples, the page reports the average follow-through, the share
that went up, and the number of distinct event dates. These are descriptive
results: related stocks and overlapping horizons are not independent samples. Under
twenty it reports the count and refuses to conclude. It makes no statistical significance claim.

That table is the whole of the agent's understanding of how the market moves.
It is a set of facts about these names over this period. It is not a model
and it does not predict.

## Connection to the trading agent

One way, and read-only. The trading agent reads `market.json` at its evening
run and records the word and the checks in its own state, so the dashboard
shows what the market agent thought that night beside what the trader did.
Nothing the market agent says changes a trade. If its calls prove out over a
few months of record, letting it gate new entries becomes a measurable
change, and it will be measured the way every other rule is.

## Commands

```
python agent.py run              read, record, publish
python agent.py run --no-news    without headlines
python agent.py run --cached     offline, from data/cache only
python agent.py page             rebuild the page from the saved read
python agent.py record           print what the record says so far
python -m tests.test_logic       the checks, offline
```

## Files

```
agent.py             the commands
mbot/config.py       the universe and every threshold
mbot/regime.py       index, breadth, volatility, sectors, the risk word
mbot/events.py       the event record and its follow-through
mbot/news.py         headline counts
mbot/report.py       the brief and the page
mbot/data.py         daily bars with a disk cache (same as the trading agent's)
state/events.jsonl   the record, committed on purpose
state/news_counts.jsonl  daily headline counts, the baseline for "a burst"
site/                the page and market.json
```

## Automatic runs and data quality

GitHub Actions runs at **3:47 p.m. America/Chicago on weekdays**, with a
**4:12 p.m. retry**. Both follow daylight saving time. The retry skips a session
already completed successfully. GitHub schedules can be delayed, so these are
scheduled start times, not guaranteed delivery times. The trader's 4:30 p.m.
Central scan reads the dated report without changing any trade decision.

An NYSE calendar handles holidays and early closes. No permanent event or
follow-through is written before the close plus 15 minutes. All inputs must
end on the same completed session; missing VIX/breadth checks are unknown.
A failure writes a visible status report and causes an unsuccessful run,
while retaining the last good reading separately. The page marks overdue
reports stale even if the scheduler stops. Weekends and holidays are included
in the report's explicit expiry time.

News counts cover unique returned Yahoo articles in the **24 hours ending at
the session close**, not all published news. The fetch requests 100 items; a
response at the limit that does not reach the window start is excluded. Each symbol
needs five valid prior observations using this method. Failed observations
are missing, never zero; the older limited-sample baseline is not reused.

Event rules are versioned. Repeated reads update same-session events and
recompute outcomes from completed bars; a missing target-session bar never
silently shifts a 1-day outcome to a different day. Tests run on code changes
and pull requests as well as before scheduled reads.

Install `requirements.txt` with Python 3.11 or newer. Runtime libraries are
pinned to tested versions. `--cached` is fully offline and skips headlines;
it still enforces current completed-session data and reports degraded status.
