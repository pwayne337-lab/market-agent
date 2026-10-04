"""
Market data loading.

Daily bars come from Yahoo Finance via yfinance. It is free and good enough
for daily swing trading research. It is not good enough for intraday work and
it will occasionally hand you a bad print, which is why load_bars sanity
checks everything before returning it.

Every download is cached to data/cache as CSV so repeated backtests do not
hammer the API and so you can work offline.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from mbot import sessions

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

# How far cached prices may sit from a fresh download of the same dates before
# the cache is treated as being on a different basis. Rounding moves prices by
# a hundredth of a percent. A split moves them by half or more.
ADJUST_TOLERANCE = 0.002
INTRADAY_RECOVERY_BUDGET = 180


class DataError(RuntimeError):
    pass


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}.csv"


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return _normalize(pd.read_csv(path, index_col=0, parse_dates=True))
    except Exception:
        return None    # an unreadable cache is not worth failing the run over


def _adjustment_drift(old: pd.DataFrame, fresh: pd.DataFrame):
    """How far apart two frames are on the dates they both cover.

    Returns (largest relative difference, number of shared bars). Zero shared
    bars means the question cannot be answered, which is itself a reason not
    to merge the two halves together.
    """
    shared = old.index.intersection(fresh.index)
    if len(shared) == 0:
        return 0.0, 0
    a = old.loc[shared, "close"].astype(float)
    b = fresh.loc[shared, "close"].astype(float)

    # Only bars where both sides have a real price can be compared. Series.max()
    # skips NaN, so an overlap that is entirely NaN used to return nan, and
    # nan > tolerance is False, so the two halves were joined anyway and the
    # NaN rows overwrote good cached bars on the way through.
    usable = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    n = int(usable.sum())
    if n == 0:
        return float("inf"), 0
    a, b = a[usable], b[usable]
    denom = b.abs().clip(lower=1e-9)
    return float(((a - b).abs() / denom).max()), n


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Force whatever the source hands back into a clean lowercase OHLCV frame."""
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns a MultiIndex when given a list of tickers.
        df = df.droplevel(1, axis=1)

    df = df.rename(columns={c: str(c).strip().lower().replace(" ", "_") for c in df.columns})

    if "adj_close" in df.columns and "close" not in df.columns:
        df["close"] = df["adj_close"]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise DataError(f"data source is missing columns: {missing}")

    df = df[REQUIRED_COLS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _sanity_check(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot be real. Bad ticks make fake backtest profits."""
    n_before = len(df)
    original = df.copy()

    # NaN checks alone miss infinity, and vendor placeholders can arrive as
    # strings. Neither is a usable price or volume, even if OHLC ordering holds.
    df = df.copy()
    df[REQUIRED_COLS] = df[REQUIRED_COLS].apply(pd.to_numeric, errors="coerce")
    df = df[np.isfinite(df[REQUIRED_COLS].to_numpy(dtype=float)).all(axis=1)]
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df = df[df["volume"] >= 0]
    # High must be the highest and low the lowest of the bar.
    df = df[df["high"] >= df[["open", "close", "low"]].max(axis=1) - 1e-9]
    df = df[df["low"] <= df[["open", "close", "high"]].min(axis=1) + 1e-9]

    dropped = n_before - len(df)
    if dropped > 0:
        rejected = original.loc[~original.index.isin(df.index)]
        latest = rejected.tail(1)
        detail = latest.to_dict(orient="index")
        print(f"  [{symbol}] dropped {dropped} malformed bar(s); latest rejected OHLCV: {detail}")
    return df


def download_bars(symbol: str, start: str, end: Optional[str] = None,
                  retries: int = 3) -> pd.DataFrame:
    """Fetch daily bars from Yahoo. Requires network access."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataError(
            "yfinance is not installed. Run: pip install -r requirements.txt"
        ) from exc

    last_err = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                symbol,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,   # split and dividend adjusted, which is what
                                    # you want for a multi-year backtest
                threads=False,
            )
            if raw is not None and len(raw) > 0:
                return _normalize(raw)
            last_err = DataError(f"empty response for {symbol}")
        except Exception as exc:  # network flake, rate limit, schema change
            last_err = exc
        time.sleep(1.5 * (attempt + 1))

    raise DataError(f"could not download {symbol}: {last_err}")


def download_many(symbols: List[str], start: str, end: Optional[str] = None,
                  chunk: int = 40) -> Dict[str, pd.DataFrame]:
    """Load groups of symbols and split yfinance's combined response.

    yfinance still requests each ticker separately. Chunking limits the size
    of each combined frame; it does not reduce the number of HTTP requests.

    Anything the batch does not return is retried on its own, because one bad
    ticker in a chunk should not cost you the other thirty-nine.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataError(
            "yfinance is not installed. Run: pip install -r requirements.txt"
        ) from exc

    out: Dict[str, pd.DataFrame] = {}
    todo = [s.upper() for s in symbols]

    for i in range(0, len(todo), chunk):
        group = todo[i:i + chunk]
        try:
            raw = yf.download(group, start=start, end=end, progress=False,
                              auto_adjust=True, threads=False, group_by="column")
        except Exception:
            continue
        if raw is None or len(raw) == 0:
            continue
        if not isinstance(raw.columns, pd.MultiIndex):
            # Yahoo flattens the columns when only one symbol comes back.
            if len(group) == 1:
                try:
                    out[group[0]] = _normalize(raw)
                except DataError:
                    pass
            continue
        for sym in group:
            try:
                one = raw.xs(sym, axis=1, level=1).dropna(how="all")
            except (KeyError, IndexError):
                continue
            one = one.dropna(subset=[c for c in ("Close", "close") if c in one.columns])
            if len(one) == 0:
                continue
            try:
                out[sym] = _normalize(one)
            except DataError:
                pass

    # Whatever the batch missed, ask for individually before giving up on it.
    for sym in todo:
        if sym in out:
            continue
        try:
            out[sym] = download_bars(sym, start=start, end=end, retries=2)
        except Exception:
            pass
    return out


def load_bars(symbol: str, start: str = "2015-01-01", end: Optional[str] = None,
              use_cache: bool = True, refresh: bool = False,
              fresh: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Load daily bars, preferring the local cache.

    Set refresh=True to force a re-download (do this before live scanning so
    you are not trading off stale bars).
    """
    symbol = symbol.upper()
    path = _cache_path(symbol)

    if use_cache and path.exists() and not refresh and fresh is None:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = _normalize(df)
    else:
        # `fresh` lets a caller hand over bars it already fetched in a batch.
        # Everything below, including the re-adjustment check, is identical
        # either way, so batching cannot quietly skip a safety step.
        if fresh is None:
            fresh = download_bars(symbol, start=start, end=end)
        fresh = _sanity_check(symbol, _normalize(fresh))

        # Merge into whatever is already cached instead of replacing it.
        # A daily run only asks for a couple of years of bars, and without
        # this merge each run would quietly delete the older history, so a
        # backtest run tomorrow would silently cover a shorter period than
        # the same command covered today. Data disappearing without an error
        # is the worst kind of bug: nothing breaks, the answers just change.
        #
        # Merging is only safe while both halves are quoted on the same basis.
        # These bars are split and dividend adjusted, so every split and every
        # dividend rewrites the entire history at the source. Gluing yesterday's
        # cache onto today's download after a 10-for-1 split would leave a 90%
        # cliff in the middle of the series, and nothing downstream would call
        # that an error. The moving averages would simply be wrong, the trend
        # filter would read a crash, and the agent would act on it. So the two
        # halves are compared where they overlap before they are joined.
        old = _read_cache(path) if use_cache else None
        if old is not None:
            old = _sanity_check(symbol, old)
        if old is not None and len(old):
            drift, shared = _adjustment_drift(old, fresh)
            if shared == 0:
                reason = "the cache and the download share no dates"
            elif drift > ADJUST_TOLERANCE:
                reason = (f"prices differ by {drift * 100:.1f}% on {shared} "
                          f"shared bar(s), the source has re-adjusted")
            else:
                reason = ""

            if reason:
                # Do not repair the seam, refuse to create one. Re-download the
                # whole history so every bar comes from one adjustment basis.
                print(f"  [{symbol}] rebuilding cache: {reason}")
                floor = str(old.index.min().date())
                begin = min(floor, start) if start else floor
                # No end date on a rebuild. The cache is about to be replaced
                # wholesale, so asking for a window that stops short of what is
                # already on disk deletes the rest of the history, and a
                # backtest run tomorrow silently covers less ground than the
                # same command covered today.
                fresh = _sanity_check(symbol, download_bars(symbol, start=begin, end=None))
            else:
                fresh = pd.concat([old, fresh])
                fresh = fresh[~fresh.index.duplicated(keep="last")].sort_index()

        df = fresh
        if df.empty:
            raise DataError(f"no usable downloaded bars for {symbol}")
        df.to_csv(path)

    df = _sanity_check(symbol, df)

    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]

    if df.empty:
        raise DataError(f"no usable bars for {symbol} in the requested range")
    return df


def load_universe(symbols: List[str], start: str = "2015-01-01",
                  end: Optional[str] = None, refresh: bool = False
                  ) -> Dict[str, pd.DataFrame]:
    """Load bars for a whole watchlist. Symbols that fail are skipped loudly."""
    out: Dict[str, pd.DataFrame] = {}

    prefetched: Dict[str, pd.DataFrame] = {}
    if refresh and len(symbols) > 1:
        prefetched = download_many(symbols, start=start, end=end)
        missed = [s for s in symbols if s.upper() not in prefetched]
        if missed:
            print(f"  {len(missed)} symbol(s) returned nothing: "
                  f"{', '.join(missed[:10])}")

    for sym in symbols:
        try:
            out[sym] = load_bars(sym, start=start, end=end, refresh=refresh,
                                 fresh=prefetched.get(sym.upper()))
        except Exception as exc:
            print(f"  [{sym}] SKIPPED: {exc}")
    if not out:
        raise DataError(
            "no symbols loaded. If every symbol failed, you have no network "
            "access to Yahoo Finance from this machine."
        )
    return out


def cache_status() -> pd.DataFrame:
    """What is in the cache and how stale it is."""
    rows = []
    for path in sorted(CACHE_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            rows.append({
                "symbol": path.stem,
                "bars": len(df),
                "first": df.index.min().date() if len(df) else None,
                "last": df.index.max().date() if len(df) else None,
                "size_kb": round(os.path.getsize(path) / 1024, 1),
            })
        except Exception:
            rows.append({"symbol": path.stem, "bars": 0, "first": None,
                         "last": None, "size_kb": 0})
    return pd.DataFrame(rows)


def _regular_session_bar(symbol: str, raw: pd.DataFrame, day: str,
                         market_open, market_close) -> pd.DataFrame:
    """Aggregate only an exact, valid set of regular-session 30-minute bars.

    Do not use _normalize here: it removes timezone information and silently
    deduplicates timestamps, both of which would hide incomplete source data.
    """
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise DataError(f"no intraday bars for {symbol}")
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        if frame.columns.nlevels != 2 or set(frame.columns.get_level_values(1)) != {symbol}:
            raise DataError(f"unexpected intraday symbols for {symbol}")
        frame = frame.droplevel(1, axis=1)
    frame = frame.rename(columns={c: str(c).strip().lower().replace(" ", "_") for c in frame.columns})
    if frame.columns.duplicated().any() or not set(REQUIRED_COLS).issubset(frame.columns):
        raise DataError(f"invalid intraday columns for {symbol}")
    frame = frame[REQUIRED_COLS]
    try:
        stamps = pd.DatetimeIndex(pd.to_datetime(frame.index))
    except (TypeError, ValueError) as exc:
        raise DataError(f"invalid intraday timestamps for {symbol}") from exc
    if stamps.tz is None or stamps.hasnans:
        raise DataError(f"intraday timestamps for {symbol} must have a timezone")
    frame.index = stamps.tz_convert("UTC")
    market_open, market_close = pd.Timestamp(market_open), pd.Timestamp(market_close)
    expected = pd.date_range(market_open, market_close, freq="30min", inclusive="left")
    regular = frame.loc[(frame.index >= market_open) & (frame.index < market_close)].sort_index()
    if regular.index.has_duplicates or not regular.index.equals(expected):
        raise DataError(f"incomplete or duplicate regular-session intraday bars for {symbol} on {day} "
                        f"({len(regular)} returned; {len(expected)} expected)")
    checked = _sanity_check(symbol, regular)
    if len(checked) != len(expected):
        raise DataError(f"malformed regular-session intraday bars for {symbol} on {day}")
    bar = pd.DataFrame({
        "open": [checked["open"].iloc[0]],
        "high": [checked["high"].max()],
        "low": [checked["low"].min()],
        "close": [checked["close"].iloc[-1]],
        "volume": [checked["volume"].sum()],
    }, index=pd.DatetimeIndex([pd.Timestamp(day)], name="date"))
    if len(_sanity_check(symbol, bar)) != 1:
        raise DataError(f"invalid aggregated intraday bar for {symbol} on {day}")
    return bar


def _recover_session_intraday(symbol: str, history: Optional[pd.DataFrame],
                              session: str) -> pd.DataFrame:
    """Recover a missing daily bar from complete, compatible intraday data.

    This is one bounded request. Both the prior and requested NYSE sessions
    must have every regular-session slot. Prior-session adjusted OHLC must
    agree with the validated daily history; no price rescaling is inferred.
    The reconstructed row stays in memory so the disk cache remains daily
    source data and a later run cannot lose the reconstruction provenance.
    """
    if symbol.startswith("^"):
        # Index publication sessions (including VIX) differ from NYSE equity
        # hours. Their daily values cannot be reconstructed on this calendar.
        raise DataError(f"intraday recovery does not support index sessions for {symbol}")
    calendar = sessions.schedule(str((pd.Timestamp(session) - pd.Timedelta(days=15)).date()), session)
    if len(calendar) < 2 or str(calendar.index[-1].date()) != session:
        raise DataError(f"no preceding trading session for {session}")
    anchor = str(calendar.index[-2].date())
    if history is None or pd.Timestamp(anchor) not in history.index:
        raise DataError(f"no validated daily adjustment anchor for {symbol} on {anchor}")
    daily_anchor = _sanity_check(symbol, history.loc[[pd.Timestamp(anchor)]])
    if len(daily_anchor) != 1:
        raise DataError(f"invalid daily adjustment anchor for {symbol} on {anchor}")

    import yfinance as yf
    raw = yf.download(symbol, start=anchor,
                      end=str((pd.Timestamp(session) + pd.Timedelta(days=1)).date()),
                      interval="30m", auto_adjust=True, prepost=False,
                      ignore_tz=False, progress=False, threads=False,
                      group_by="column", timeout=15)
    intraday_anchor, recovered = [
        _regular_session_bar(symbol, raw, day, calendar.loc[day, "market_open"],
                             calendar.loc[day, "market_close"])
        for day in (anchor, session)
    ]
    prices = REQUIRED_COLS[:4]
    reference = daily_anchor[prices].iloc[0].astype(float)
    observed = intraday_anchor[prices].iloc[0].astype(float)
    drift = ((reference - observed).abs() / reference).max()
    if not np.isfinite(drift) or drift > ADJUST_TOLERANCE:
        raise DataError(f"intraday adjustment anchor differs for {symbol} on {anchor} "
                        f"({drift * 100:.3f}%; maximum {ADJUST_TOLERANCE * 100:.3f}%)")
    result = pd.concat([history.loc[history.index != pd.Timestamp(session)], recovered]).sort_index()
    result.attrs["price_recovery"] = {
        "provider": "Yahoo Finance", "as_of": session,
        "method": "complete_regular_session_30m", "interval": "30m",
        "anchor_session": anchor,
        "bars": len(pd.date_range(calendar.loc[session, "market_open"],
                                  calendar.loc[session, "market_close"],
                                  freq="30min", inclusive="left")),
    }
    return result


def load_session(symbols: List[str], start: str, session: str,
                 cached: bool = False) -> Dict[str, pd.DataFrame]:
    """Read a completed session, retrying nonempty but stale batch responses.

    Yahoo's end date is exclusive. Supplying it explicitly also makes a late
    run after UTC midnight request the same completed New York session.
    Cached mode never downloads, including when a cached symbol is stale.
    """
    end = str((pd.Timestamp(session) + pd.Timedelta(days=1)).date())
    valid_dates = sessions.schedule(start, session).index
    if pd.Timestamp(session) not in valid_dates:
        raise DataError(f"{session} is not a trading session")
    selected = [s for s in symbols if _cache_path(s).exists()] if cached else symbols
    try:
        bars = load_universe(selected, start=start, end=end, refresh=not cached)
    except DataError as exc:
        # Even a wholly unusable batch gets the independent recovery window.
        # Missing inputs still fail the report if recovery does not succeed.
        print(f"  Session batch unavailable: {exc}")
        bars = {}

    def session_bars(symbol, frame):
        if frame is None:
            return None
        selected = frame.loc[frame.index.isin(valid_dates)]
        excluded = len(frame) - len(selected)
        if excluded:
            print(f"  [{symbol}] excluded {excluded} bar(s) outside requested trading sessions")
        return selected

    bars = {sym: session_bars(sym, frame) for sym, frame in bars.items()}

    def current(frame):
        return frame is not None and pd.Timestamp(session) in frame.index

    recovery_waited = False
    for sym in symbols:
        frame = bars.get(sym)
        if current(frame):
            continue
        last = str(frame.index[-1].date()) if frame is not None and len(frame) else "none"
        print(f"  [{sym}] session {session} missing (latest usable bar: {last})")
        if cached:
            continue
        # Nonempty stale results used to bypass all download retries. Retry
        # independently, then validate again; yesterday's prices never qualify.
        for attempt in range(2):
            try:
                if attempt == 0:
                    frame = load_bars(sym, start=start, end=end, refresh=True)
                else:
                    # A different request window avoids repeating the same bad
                    # long-history response. Keep >201 trading days and let
                    # load_bars enforce adjustment compatibility with the cache.
                    recovery_start = max(start, str((pd.Timestamp(session) - pd.Timedelta(days=550)).date()))
                    if recovery_start == start:
                        recovery_start = str((pd.Timestamp(start) - pd.Timedelta(days=7)).date())
                    if not recovery_waited:
                        time.sleep(5)
                        recovery_waited = True
                    fresh = download_bars(sym, start=recovery_start, end=end)
                    frame = load_bars(sym, start=start, end=end, refresh=True, fresh=fresh)
                frame = session_bars(sym, frame)
                bars[sym] = frame
                if current(frame):
                    print(f"  [{sym}] recovered session {session} on retry {attempt + 1}")
                    break
            except Exception as exc:
                print(f"  [{sym}] session retry {attempt + 1} failed: {exc}")
    if not cached:
        recovery_started = time.monotonic()
        missing = [sym for sym in symbols if not current(bars.get(sym)) and not sym.startswith("^")]
        # Recover the essential broad index ETF first during a broad feed failure.
        missing.sort(key=lambda sym: sym != "SPY")
        for sym in missing:
            if time.monotonic() - recovery_started + 15 > INTRADAY_RECOVERY_BUDGET:
                print("  Intraday recovery time budget exhausted; remaining symbols stay unavailable")
                break
            try:
                recovered = _recover_session_intraday(sym, bars.get(sym), session)
                bars[sym] = session_bars(sym, recovered)
                print(f"  [{sym}] recovered session {session} from complete regular-session 30-minute bars")
            except Exception as exc:
                print(f"  [{sym}] intraday recovery unavailable: {exc}")
                bars.pop(sym, None)
    return {s: df.loc[df.index <= session] for s, df in bars.items() if current(df)}
