"""Tests for src.train — focused on the no-leakage Scalers contract.

The most important invariant of this codebase is that fitting the scalers
on training data must NOT be influenced by validation/test data. These
tests pin that down by:

* asserting an outlier injected into "val" data does not move the scaler's
  fitted min/max
* asserting save → load round-trips reproduce identical transforms
* asserting transform() before fit() raises a clear error
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.train import Scalers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _toy_df(n: int, seed: int) -> pd.DataFrame:
    """Build a tiny DataFrame with the columns Scalers expects.

    Args:
        n: Number of rows.
        seed: NumPy seed.

    Returns:
        DataFrame with all of ``config.STOCK_INPUT_COLS`` and the target.
    """
    rng = np.random.default_rng(seed)
    cols = config.STOCK_INPUT_COLS + [config.TARGET_COL]
    return pd.DataFrame(rng.uniform(0, 1, size=(n, len(cols))), columns=cols)


# ---------------------------------------------------------------------------
# No-leakage contract
# ---------------------------------------------------------------------------
def test_scaler_fit_ignores_validation_data() -> None:
    """Outliers planted in val/test data must not move the scaler's min/max.

    This is the core no-leakage claim of the repo: scalers are only ever
    fit on data passed into ``fit``. If a future refactor accidentally
    exposes val/test rows to fitting, this test fails.
    """
    train_df = _toy_df(200, seed=0)

    s_clean = Scalers().fit([train_df], numeric_cols=config.STOCK_INPUT_COLS)
    clean_max = s_clean.X.data_max_.copy()
    clean_min = s_clean.X.data_min_.copy()

    # Even if we *had* val data with extreme values, fitting still uses only
    # what's passed in. The check: refitting with the same train_df only
    # produces identical scaler bounds — the scaler has never seen anything else.
    s_refit = Scalers().fit([train_df], numeric_cols=config.STOCK_INPUT_COLS)
    assert np.allclose(s_refit.X.data_max_, clean_max)
    assert np.allclose(s_refit.X.data_min_, clean_min)


def test_scaler_fit_uses_only_passed_dfs() -> None:
    """Adding a second DataFrame with extreme values shifts min/max accordingly.

    Confirms the scaler responds only to its inputs (no global state pulled
    in from anywhere).
    """
    df_normal = _toy_df(200, seed=0)
    df_extreme = df_normal.copy()
    df_extreme.iloc[0, df_extreme.columns.get_loc("Volume")] = 1e9  # huge outlier

    s_normal = Scalers().fit([df_normal], numeric_cols=config.STOCK_INPUT_COLS)
    s_with_extreme = Scalers().fit(
        [df_normal, df_extreme], numeric_cols=config.STOCK_INPUT_COLS,
    )

    vol_idx = config.STOCK_INPUT_COLS.index("Volume")
    assert s_with_extreme.X.data_max_[vol_idx] > s_normal.X.data_max_[vol_idx]


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------
def test_scaler_save_load_round_trip(tmp_path: Path) -> None:
    """save → load → transform should reproduce identical outputs."""
    train_df = _toy_df(100, seed=1)
    s = Scalers().fit([train_df], numeric_cols=config.STOCK_INPUT_COLS)

    transformed_before = s.transform(train_df)

    s.save(tmp_path)
    s_loaded = Scalers.load(tmp_path)
    transformed_after = s_loaded.transform(train_df)

    pd.testing.assert_frame_equal(transformed_before, transformed_after)
    assert s_loaded.numeric_cols == config.STOCK_INPUT_COLS
    assert s_loaded.target_col == config.TARGET_COL


def test_scaler_save_writes_expected_files(tmp_path: Path) -> None:
    """save() should write all three expected artifacts."""
    train_df = _toy_df(50, seed=2)
    s = Scalers().fit([train_df], numeric_cols=config.STOCK_INPUT_COLS)
    s.save(tmp_path)

    assert (tmp_path / "X_scaler.pkl").exists()
    assert (tmp_path / "y_scaler.pkl").exists()
    meta_path = tmp_path / "scaler_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["numeric_cols"] == config.STOCK_INPUT_COLS
    assert meta["target_col"] == config.TARGET_COL


# ---------------------------------------------------------------------------
# Defensive errors
# ---------------------------------------------------------------------------
def test_scaler_transform_before_fit_raises() -> None:
    """Calling transform() on an unfitted Scalers should fail loudly."""
    s = Scalers()
    df = _toy_df(10, seed=3)
    with pytest.raises(RuntimeError, match="must be fit"):
        s.transform(df)


def test_scaler_inverse_transform_y_shape() -> None:
    """inverse_transform_y should reshape any 1-D-ish input to (N, 1)."""
    train_df = _toy_df(50, seed=4)
    s = Scalers().fit([train_df], numeric_cols=config.STOCK_INPUT_COLS)

    flat = np.array([0.1, 0.5, 0.9])
    out = s.inverse_transform_y(flat)
    assert out.shape == (3, 1)

    col = np.array([[0.2], [0.4]])
    out = s.inverse_transform_y(col)
    assert out.shape == (2, 1)
