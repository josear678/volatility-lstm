"""Tests for src.evaluate — metric correctness and naive-baseline mechanics."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import compute_metrics, naive_baseline


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------
def test_compute_metrics_perfect_prediction() -> None:
    """If y_pred == y_true, RMSE/MAE/MAPE should be 0 and R² should be 1."""
    y = np.array([0.1, 0.2, 0.3, 0.4]).reshape(-1, 1)
    m = compute_metrics(y, y)
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["mape"] == 0.0
    assert m["r2"] == pytest.approx(1.0)


def test_compute_metrics_known_values() -> None:
    """Compare against hand-computed values for a tiny known input."""
    y_true = np.array([1.0, 2.0, 3.0]).reshape(-1, 1)
    y_pred = np.array([1.5, 1.5, 3.5]).reshape(-1, 1)

    # By hand:
    #   errors = [-0.5, 0.5, -0.5]   |errors| = [0.5, 0.5, 0.5]
    #   MSE    = (0.25 + 0.25 + 0.25) / 3 = 0.25
    #   RMSE   = 0.5
    #   MAE    = 0.5
    m = compute_metrics(y_true, y_pred)
    assert m["rmse"] == pytest.approx(0.5)
    assert m["mae"] == pytest.approx(0.5)
    # R² > 0 because the predictions are better than predicting the mean
    assert 0.0 < m["r2"] < 1.0


def test_compute_metrics_returns_floats() -> None:
    """All metric values must be plain Python floats (JSON-serializable)."""
    y = np.array([0.0, 1.0, 2.0]).reshape(-1, 1)
    m = compute_metrics(y, y + 0.1)
    for key in ("rmse", "mae", "r2", "mape"):
        assert isinstance(m[key], float), f"{key} should be float, got {type(m[key])}"


# ---------------------------------------------------------------------------
# naive_baseline
# ---------------------------------------------------------------------------
def test_naive_baseline_extracts_last_timestep() -> None:
    """The baseline value should be the last timestep of the chosen feature."""
    # Shape (N=2, window_size=3, features=2). Feature 0 is the baseline.
    X = np.array([
        [[0.1, 9.0], [0.2, 9.0], [0.3, 9.0]],   # baseline (last) = 0.3
        [[0.5, 9.0], [0.6, 9.0], [0.7, 9.0]],   # baseline (last) = 0.7
    ])
    out = naive_baseline(X, input_cols=["target_feat", "other"], baseline_feature="target_feat")
    assert out.shape == (2, 1)
    assert np.allclose(out.flatten(), [0.3, 0.7])


def test_naive_baseline_unknown_feature_raises() -> None:
    """Asking for a feature that doesn't exist should fail clearly."""
    X = np.zeros((1, 3, 2))
    with pytest.raises(ValueError, match="not in input_cols"):
        naive_baseline(X, input_cols=["a", "b"], baseline_feature="missing")
