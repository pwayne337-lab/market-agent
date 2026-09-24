"""NYSE session boundaries, including holidays, DST and early closes."""
from datetime import datetime, timezone
from functools import lru_cache
import pandas as pd
import pandas_market_calendars as mcal

@lru_cache(maxsize=32)
def schedule(start, end):
    return mcal.get_calendar('NYSE').schedule(start_date=start, end_date=end)

def utc_now():
    return datetime.now(timezone.utc)

def today_session(now=None):
    stamp = pd.Timestamp(now or utc_now())
    stamp = stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')
    day = str(stamp.tz_convert('America/New_York').date())
    cal = schedule(day, day)
    if cal.empty:
        return None
    close = cal.iloc[0]['market_close']
    return {'date': day, 'close': close, 'ready': stamp >= close + pd.Timedelta(minutes=15)}

def last_completed(now=None):
    stamp = pd.Timestamp(now or utc_now())
    stamp = stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')
    cal = schedule(str((stamp - pd.Timedelta(days=15)).date()), str(stamp.date()))
    ready = cal[cal.market_close + pd.Timedelta(minutes=15) <= stamp]
    return str(ready.index[-1].date())

def next_expiry(day):
    cal = schedule(day, str((pd.Timestamp(day) + pd.Timedelta(days=15)).date()))
    future = cal[cal.index > pd.Timestamp(day)]
    # Allow the next scheduled read and retry before marking a prior report stale.
    return (future.iloc[0].market_close + pd.Timedelta(minutes=90)).isoformat()

def follow_dates(day, horizon):
    cal = schedule(day, str((pd.Timestamp(day) + pd.Timedelta(days=max(30, horizon * 3))).date()))
    dates = [str(d.date()) for d in cal.index if d > pd.Timestamp(day)]
    return dates[horizon - 1]
