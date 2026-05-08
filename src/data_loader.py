"""Fetch OHLC data and VIX from yfinance, with optional parquet caching.

Pure I/O — no feature engineering happens here. Caching avoids hammering Yahoo
and keeps demos working offline once the cache is populated.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _safe_filename(ticker: str) -> str:
    """Convert a Yahoo ticker into a filesystem-safe filename stem.

    Yahoo tickers like ``^VIX`` and ``^GSPC`` start with a caret which most
    shells interpret specially; some cross-platform filesystems also dislike
    forward and back slashes.

    Args:
        ticker: The Yahoo ticker symbol (e.g. ``"^VIX"``, ``"BRK-B"``).

    Returns:
        A filesystem-safe filename stem (e.g. ``"VIX"``, ``"BRK-B"``).
    """
    return ticker.replace("^", "").replace("/", "_").replace("\\", "_")


def _cache_path(ticker: str, cache_dir: Path) -> Path:
    """Build the parquet cache path for a given ticker.

    Args:
        ticker: The Yahoo ticker symbol.
        cache_dir: Directory in which cache files live.

    Returns:
        Full path to the ticker's parquet cache file.
    """
    return Path(cache_dir) / f"{_safe_filename(ticker)}.parquet"


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a tz-naive midnight-normalized DatetimeIndex.

    yfinance returns tz-aware indexes (typically America/New_York). Stripping
    the timezone and normalizing to midnight produces a clean calendar-date
    index that joins cleanly with other date-indexed series (e.g. VIX).

    Args:
        df: A DataFrame with a DatetimeIndex.

    Returns:
        A copy of ``df`` with the index reset to tz-naive midnight timestamps.
    """
    df = df.copy()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_ohlc(
    ticker: str,
    period: str = "max",
    interval: str = "1d",
    cache_dir: Path | str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch a ticker's OHLC history from yfinance, optionally cached to parquet.

    On first call with ``cache_dir`` set, the result is written to a parquet
    file. Subsequent calls read from disk in <1 ms instead of hitting the
    network.

    Args:
        ticker: Yahoo ticker symbol (e.g. ``"AAPL"``, ``"^VIX"``).
        period: yfinance period string (``"max"``, ``"5y"``, etc.). Defaults
            to ``"max"``.
        interval: yfinance bar interval (``"1d"``, ``"1h"``, etc.). Defaults
            to ``"1d"``.
        cache_dir: If provided, results are read from / written to parquet
            files under this directory. ``None`` disables caching.
        force_refresh: If ``True``, ignore any existing cache entry and
            re-fetch from yfinance. Defaults to ``False``.

    Returns:
        A DataFrame of OHLC bars with a tz-naive DatetimeIndex.

    Raises:
        ValueError: If yfinance returns an empty DataFrame for ``ticker``.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else None

    # Cache-hit path
    if cache_dir is not None and not force_refresh:
        p = _cache_path(ticker, cache_dir)
        if p.exists():
            logger.info("Loading %s from cache: %s", ticker, p)
            df = pd.read_parquet(p)
            return _normalize_index(df)

    # Cache miss — hit yfinance
    logger.info("Fetching %s from yfinance (period=%s, interval=%s)", ticker, period, interval)
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker!r}")
    df = _normalize_index(df)

    # Populate cache for next time
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        p = _cache_path(ticker, cache_dir)
        df.to_parquet(p)
        logger.info("Cached %s to %s", ticker, p)

    return df


def fetch_many(
    tickers: Iterable[str],
    cache_dir: Path | str | None = None,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch multiple tickers' OHLC histories.

    Args:
        tickers: Iterable of Yahoo ticker symbols.
        cache_dir: Optional cache directory; passed through to :func:`fetch_ohlc`.
        force_refresh: If ``True``, bypass the cache for every ticker.

    Returns:
        Dictionary mapping each ticker to its OHLC DataFrame.
    """
    return {
        t: fetch_ohlc(t, cache_dir=cache_dir, force_refresh=force_refresh)
        for t in tickers
    }


def fetch_vix(
    period: str = "max",
    cache_dir: Path | str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch the VIX index history.

    The VIX (CBOE Volatility Index) is used as an exogenous market-stress
    feature. Only its closing level is retained; other OHLC columns are
    discarded.

    Args:
        period: yfinance period string. Defaults to ``"max"``.
        cache_dir: Optional cache directory passed to :func:`fetch_ohlc`.
        force_refresh: If ``True``, bypass the cache.

    Returns:
        A DataFrame with a single ``vix_close`` column indexed by date.
    """
    df = fetch_ohlc(
        config.VIX_TICKER, period=period, cache_dir=cache_dir, force_refresh=force_refresh
    )
    return df[["Close"]].rename(columns={"Close": "vix_close"})
