"""Deterministic tests for src.features.

These run on synthetic OHLC and VIX data so they need no network access and
finish in <1 second. They cover:

* per-column feature correctness
* shift alignment of the target
* VIX join column presence
* engineer_features end-to-end shape and absence of NaNs
* sliding-window shapes and alignment
* chronological-split proportions and lossless reconstruction
* defensive ValueError when the input is too short to window
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, features


# ---------------------------------------------------------------------------
# Fixtures (synthetic; deterministic via seed)
# ---------------------------------------------------------------------------
def make_synthetic_ohlc(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Build a deterministic synthetic OHLC frame for testing.

    Generates a random-walk close series with corresponding high / low /
    volume. Enough non-trivial variation for rolling-window features to
    produce non-degenerate values.

    Args:
        n: Number of trading-day rows to generate.
        seed: NumPy random seed for reproducibility.

    Returns:
        A DataFrame with columns ``Open``, ``High``, ``Low``, ``Close``,
        ``Volume`` and a business-day DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n)
    # Random-walk close so volatility features are non-degenerate
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 1, n))
    low = close - np.abs(rng.normal(0, 1, n))
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def make_synthetic_vix(n: int = 400, seed: int = 1) -> pd.DataFrame:
    """Build a deterministic synthetic VIX series for testing.

    Args:
        n: Number of trading-day rows to generate.
        seed: NumPy random seed for reproducibility.

    Returns:
        A DataFrame with a single ``vix_close`` column and a business-day
        DatetimeIndex; values are noisy around 15 (a typical VIX level).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame(
        {"vix_close": 15 + np.abs(rng.normal(0, 3, n))},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Per-feature unit tests
# ---------------------------------------------------------------------------
def test_add_moving_averages_columns_present() -> None:
    """``add_moving_averages`` should add one column per requested window."""
    out = features.add_moving_averages(make_synthetic_ohlc(), windows=(20, 50))
    assert "MA_20" in out.columns
    assert "MA_50" in out.columns


def test_add_moving_averages_correct_value() -> None:
    """First fully-populated MA value should equal the simple mean of its window."""
    df = make_synthetic_ohlc(n=100)
    out = features.add_moving_averages(df, windows=(5,))
    expected = df["Close"].iloc[:5].mean()
    assert np.isclose(out["MA_5"].iloc[4], expected)


def test_add_returns_first_value_is_nan_and_columns() -> None:
    """``return`` is NaN at row 0 (no prior close); ``daily_range`` is non-negative."""
    df = make_synthetic_ohlc()
    out = features.add_returns(df)
    assert pd.isna(out["return"].iloc[0])
    assert "daily_range" in out.columns
    assert (out["daily_range"].dropna() >= 0).all()


def test_add_volatility_features_columns_present() -> None:
    """``add_volatility_features`` should add all four volatility columns."""
    df = features.add_returns(make_synthetic_ohlc())
    out = features.add_volatility_features(df)
    for col in (
        "daily_volatility",
        "weekly_volatility_fast",
        "weekly_volatility_slow",
        "yearly_volatility",
    ):
        assert col in out.columns


def test_add_volatility_features_weekly_fast_is_daily_times_sqrt5() -> None:
    """The weekly_fast = daily × √5 annualization identity should hold per row."""
    df = features.add_returns(make_synthetic_ohlc())
    out = features.add_volatility_features(df)
    # Pick a row beyond the rolling window so values are populated
    i = 50
    assert np.isclose(
        out["weekly_volatility_fast"].iloc[i],
        out["daily_volatility"].iloc[i] * np.sqrt(5),
    )


def test_add_target_shift_aligns() -> None:
    """Row i's target should equal weekly_volatility_fast at row i+horizon."""
    df = features.add_returns(make_synthetic_ohlc())
    df = features.add_volatility_features(df)
    out = features.add_target(df, horizon=5)
    assert np.isclose(
        out["target_volatility"].iloc[10],
        out["weekly_volatility_fast"].iloc[15],
        equal_nan=True,
    )


def test_add_vix_features_columns_present() -> None:
    """``add_vix_features`` should produce all four VIX-derived columns."""
    out = features.add_vix_features(make_synthetic_ohlc(n=100), make_synthetic_vix(n=100))
    for col in ("vix_close", "vix_return", "vix_ma_20", "vix_rolling_std_20"):
        assert col in out.columns


# ---------------------------------------------------------------------------
# Pipeline-level tests
# ---------------------------------------------------------------------------
def test_engineer_features_with_vix_no_nans() -> None:
    """End-to-end pipeline must drop all NaN rows and contain every input col."""
    out = features.engineer_features(make_synthetic_ohlc(), make_synthetic_vix())
    assert not out.isna().any().any(), "engineer_features should drop all NaN rows"
    for c in config.INPUT_COLS + [config.TARGET_COL]:
        assert c in out.columns


def test_engineer_features_without_vix_omits_vix_cols() -> None:
    """When ``vix_df=None``, VIX columns should not appear at all."""
    out = features.engineer_features(make_synthetic_ohlc(), vix_df=None)
    for c in config.STOCK_INPUT_COLS + [config.TARGET_COL]:
        assert c in out.columns
    for c in config.VIX_INPUT_COLS:
        assert c not in out.columns


# ---------------------------------------------------------------------------
# Windowing & split
# ---------------------------------------------------------------------------
def test_make_windows_shapes() -> None:
    """X is (N - window_size, window_size, F); y is (N - window_size, 1)."""
    df = features.engineer_features(make_synthetic_ohlc(), make_synthetic_vix())
    X, y = features.make_windows(
        df, config.INPUT_COLS, config.TARGET_COL, window_size=30,
    )
    assert X.shape == (len(df) - 30, 30, len(config.INPUT_COLS))
    assert y.shape == (len(df) - 30, 1)


def test_make_windows_alignment() -> None:
    """Window i contains rows [i, i+window_size); target is row i+window_size."""
    df = features.engineer_features(make_synthetic_ohlc(), make_synthetic_vix()).reset_index(drop=True)
    X, y = features.make_windows(
        df, config.INPUT_COLS, config.TARGET_COL, window_size=30,
    )
    # First row of window 5 should match df row 5, feature 0
    assert np.isclose(X[5, 0, 0], df[config.INPUT_COLS].iloc[5, 0])
    # Last row of window 5 should match df row 5+29
    assert np.isclose(X[5, 29, 0], df[config.INPUT_COLS].iloc[5 + 29, 0])
    # Target of window 5 should match df row 5+30
    assert np.isclose(y[5, 0], df[config.TARGET_COL].iloc[5 + 30])


def test_make_windows_too_small_raises() -> None:
    """``make_windows`` should raise ValueError when df is shorter than window_size."""
    df = features.engineer_features(make_synthetic_ohlc(n=400), vix_df=None).head(10)
    with pytest.raises(ValueError):
        features.make_windows(
            df, config.STOCK_INPUT_COLS, config.TARGET_COL, window_size=30,
        )


def test_chronological_split_proportions() -> None:
    """80/10/10 split of 1000 rows should yield 800/100/100."""
    X = np.zeros((1000, 30, 13), dtype=np.float32)
    y = np.zeros((1000, 1), dtype=np.float32)
    s = features.chronological_split(X, y, train_frac=0.8, val_frac=0.1)
    assert s["X_train"].shape[0] == 800
    assert s["X_val"].shape[0] == 100
    assert s["X_test"].shape[0] == 100


def test_chronological_split_is_lossless_and_in_order() -> None:
    """Splits should reconstruct exactly and remain in chronological order."""
    X = np.arange(1000, dtype=np.float32).reshape(1000, 1, 1)
    y = np.arange(1000, dtype=np.float32).reshape(1000, 1)
    s = features.chronological_split(X, y, train_frac=0.8, val_frac=0.1)
    reconstructed_X = np.concatenate([s["X_train"], s["X_val"], s["X_test"]])
    reconstructed_y = np.concatenate([s["y_train"], s["y_val"], s["y_test"]])
    assert np.array_equal(reconstructed_X, X)
    assert np.array_equal(reconstructed_y, y)
    # train ends before val begins; val ends before test begins
    assert s["X_train"][-1, 0, 0] < s["X_val"][0, 0, 0]
    assert s["X_val"][-1, 0, 0] < s["X_test"][0, 0, 0]
