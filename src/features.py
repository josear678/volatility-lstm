"""Feature engineering: moving averages, returns, volatility, VIX, target, windowing.

Pure functions — given a DataFrame, return a DataFrame. No fitting, no I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Per-stock feature derivations
# ---------------------------------------------------------------------------
def add_moving_averages(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (20, 50),
) -> pd.DataFrame:
    """Add ``MA_<w>`` columns for each window size.

    Args:
        df: An OHLC DataFrame containing a ``Close`` column.
        windows: Iterable of integer window lengths in trading days.
            Defaults to ``(20, 50)``.

    Returns:
        A copy of ``df`` with one new column per window
        (e.g. ``MA_20``, ``MA_50``).
    """
    df = df.copy()
    for w in windows:
        df[f"MA_{w}"] = df["Close"].rolling(w).mean()
    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily-return and intraday-range columns.

    Args:
        df: An OHLC DataFrame containing ``Close``, ``High``, and ``Low``.

    Returns:
        A copy of ``df`` with two new columns:

        * ``return``       — daily percent change of ``Close``
        * ``daily_range``  — ``(High - Low) / Close`` (a normalized range)
    """
    df = df.copy()
    df["return"] = df["Close"].pct_change()
    df["daily_range"] = (df["High"] - df["Low"]) / df["Close"]
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-horizon realized-volatility columns.

    The ``return`` column must already exist (call :func:`add_returns` first).
    Annualization uses the standard ``√N`` rule (5 trading days per week,
    252 per year).

    Args:
        df: A DataFrame containing a ``return`` column.

    Returns:
        A copy of ``df`` with four new columns:

        * ``daily_volatility``        — 5-day rolling std of ``return``
        * ``weekly_volatility_fast``  — ``daily_volatility × √5``
        * ``weekly_volatility_slow``  — 20-day rolling std × √5
        * ``yearly_volatility``       — 252-day rolling std × √252
    """
    df = df.copy()
    df["daily_volatility"] = df["return"].rolling(5).std()
    df["weekly_volatility_fast"] = df["daily_volatility"] * np.sqrt(5)
    df["weekly_volatility_slow"] = df["return"].rolling(20).std() * np.sqrt(5)
    df["yearly_volatility"] = df["return"].rolling(252).std() * np.sqrt(252)
    return df


def add_vix_features(df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join VIX onto ``df`` and compute four VIX-derived features.

    Both indexes are normalized to tz-naive calendar dates before joining.
    VIX is forward-filled across any non-trading days that differ between
    the two series so the per-stock DataFrame never has gaps in the
    VIX columns.

    Args:
        df: An OHLC DataFrame to augment.
        vix_df: A DataFrame with at least a ``vix_close`` column, indexed
            by date.

    Returns:
        A copy of ``df`` with four new columns:

        * ``vix_close``            — daily VIX closing level
        * ``vix_return``           — daily percent change in VIX
        * ``vix_ma_20``            — 20-day rolling mean of VIX
        * ``vix_rolling_std_20``   — 20-day rolling std of VIX (vol-of-vol)
    """
    df = df.copy()
    vix = vix_df.copy()

    # Normalize both indexes so the join keys are calendar dates, not timestamps
    df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) else pd.to_datetime(df.index)
    df.index = df.index.normalize()
    vix.index = pd.to_datetime(vix.index).tz_localize(None) if getattr(vix.index, "tz", None) else pd.to_datetime(vix.index)
    vix.index = vix.index.normalize()

    df = df.join(vix[["vix_close"]], how="left")
    df["vix_close"] = df["vix_close"].ffill()
    df["vix_return"] = df["vix_close"].pct_change()
    df["vix_ma_20"] = df["vix_close"].rolling(20).mean()
    df["vix_rolling_std_20"] = df["vix_close"].rolling(20).std()
    return df


def add_target(
    df: pd.DataFrame,
    horizon: int = config.FORECAST_HORIZON,
) -> pd.DataFrame:
    """Add the prediction target column to ``df``.

    The target is ``weekly_volatility_fast`` shifted backward by ``horizon``
    rows, so row ``i`` holds the volatility we want to predict at row
    ``i + horizon``.

    Args:
        df: A DataFrame containing ``weekly_volatility_fast``.
        horizon: Number of trading days to look ahead. Defaults to
            ``config.FORECAST_HORIZON``.

    Returns:
        A copy of ``df`` with a new ``target_volatility`` column. The last
        ``horizon`` rows will contain NaN and are typically dropped later.
    """
    df = df.copy()
    df["target_volatility"] = df["weekly_volatility_fast"].shift(-horizon)
    return df


def engineer_features(
    df: pd.DataFrame,
    vix_df: pd.DataFrame | None = None,
    horizon: int = config.FORECAST_HORIZON,
) -> pd.DataFrame:
    """Run the full feature-engineering pipeline on a single stock.

    Calls, in order: :func:`add_moving_averages`, :func:`add_returns`,
    :func:`add_volatility_features`, optionally :func:`add_vix_features`,
    then :func:`add_target`. Drops any row that contains a NaN at the end,
    which removes the leading rows from the rolling-window features and the
    trailing rows from the target shift.

    Args:
        df: An OHLC DataFrame.
        vix_df: Optional VIX DataFrame. If ``None``, VIX features are skipped
            (use this when training a stock-only model).
        horizon: Forecast horizon in trading days, passed to :func:`add_target`.

    Returns:
        A fully-engineered DataFrame with no NaNs, ready for windowing.
    """
    df = add_moving_averages(df)
    df = add_returns(df)
    df = add_volatility_features(df)
    if vix_df is not None:
        df = add_vix_features(df, vix_df)
    df = add_target(df, horizon=horizon)
    return df.dropna()


# ---------------------------------------------------------------------------
# Windowing & chronological split
# ---------------------------------------------------------------------------
def make_windows(
    df: pd.DataFrame,
    input_cols: list[str],
    target_col: str = config.TARGET_COL,
    window_size: int = config.WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window LSTM inputs.

    Window ``i`` contains rows ``[i, i + window_size)`` of the input columns;
    its label is the target value at row ``i + window_size``.

    Args:
        df: Engineered DataFrame containing both ``input_cols`` and
            ``target_col``.
        input_cols: Column names used as model inputs. Order is preserved
            in the feature dimension.
        target_col: Column name of the prediction target.
            Defaults to ``config.TARGET_COL``.
        window_size: Number of consecutive rows per window. Defaults to
            ``config.WINDOW_SIZE``.

    Returns:
        A tuple ``(X, y)`` where:

        * ``X`` is shape ``(N, window_size, len(input_cols))`` of dtype float32
        * ``y`` is shape ``(N, 1)`` of dtype float32

        and ``N = len(df) - window_size``.

    Raises:
        ValueError: If ``df`` has fewer than ``window_size + 1`` rows.
    """
    n = len(df) - window_size
    if n <= 0:
        raise ValueError(
            f"Not enough rows ({len(df)}) for window_size={window_size}. "
            "Need at least window_size + 1 rows."
        )
    X = np.zeros((n, window_size, len(input_cols)), dtype=np.float32)
    y = np.zeros((n, 1), dtype=np.float32)
    feat = df[input_cols].values.astype(np.float32)
    targ = df[[target_col]].values.astype(np.float32)
    for i in range(n):
        X[i] = feat[i : i + window_size]
        y[i] = targ[i + window_size]
    return X, y


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
) -> dict[str, np.ndarray]:
    """Split ``(X, y)`` chronologically into train / validation / test.

    No shuffling. The earliest ``train_frac`` rows go to train, the next
    ``val_frac`` to validation, and the remainder to test.

    Args:
        X: Input array shaped ``(N, ...)``.
        y: Target array shaped ``(N, ...)``.
        train_frac: Fraction of rows to use for training.
            Defaults to ``config.TRAIN_FRAC``.
        val_frac: Fraction of rows to use for validation. Test fraction is
            the remainder, ``1 - train_frac - val_frac``.
            Defaults to ``config.VAL_FRAC``.

    Returns:
        Dictionary with keys ``X_train``, ``y_train``, ``X_val``, ``y_val``,
        ``X_test``, ``y_test``.
    """
    n = len(X)
    train_end = int(np.floor(n * train_frac))
    val_end = int(np.floor(n * (train_frac + val_frac)))
    return {
        "X_train": X[:train_end],
        "y_train": y[:train_end],
        "X_val": X[train_end:val_end],
        "y_val": y[train_end:val_end],
        "X_test": X[val_end:],
        "y_test": y[val_end:],
    }
