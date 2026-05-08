"""Metrics and plots.

All metrics are computed in real-world (inverse-transformed) units so they're
directly interpretable as volatility values.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from .train import Scalers


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the standard regression metrics for a prediction.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values, same shape as ``y_true``.

    Returns:
        Dictionary with keys ``rmse``, ``mae``, ``r2``, ``mape``.
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def naive_baseline(
    X: np.ndarray,
    input_cols: list[str],
    baseline_feature: str = "weekly_volatility_fast",
) -> np.ndarray:
    """Build the "next week ≈ this week" naive prediction.

    Returns the value of ``baseline_feature`` at the *last* timestep of each
    window. The result is still in scaled space; the caller is expected to
    inverse-transform it through the same y_scaler used by the model.

    Args:
        X: Window array of shape ``(N, window_size, num_features)``.
        input_cols: Feature names corresponding to the last axis of ``X``.
        baseline_feature: Which feature to use as the naive prediction.
            Defaults to ``"weekly_volatility_fast"`` — the same quantity
            the target is derived from, just shifted in time.

    Returns:
        Naive predictions shaped ``(N, 1)`` in scaled space.

    Raises:
        ValueError: If ``baseline_feature`` is not in ``input_cols``.
    """
    if baseline_feature not in input_cols:
        raise ValueError(
            f"baseline_feature {baseline_feature!r} not in input_cols={input_cols}"
        )
    idx = input_cols.index(baseline_feature)
    return X[:, -1, idx].reshape(-1, 1)


def evaluate_on_split(
    model,
    X: np.ndarray,
    y: np.ndarray,
    scalers: "Scalers",
    input_cols: list[str],
    baseline_feature: str = "weekly_volatility_fast",
) -> dict:
    """Evaluate the model and naive baseline on a single split.

    Inverse-transforms both predictions and targets so all reported metrics
    are in real-world volatility units.

    Args:
        model: Trained Keras model.
        X: Input windows for this split.
        y: Targets for this split, in scaled space.
        scalers: Fitted scaler bundle from training.
        input_cols: Feature names matching the last axis of ``X``.
        baseline_feature: Feature used by the naive baseline. Defaults to
            ``"weekly_volatility_fast"``.

    Returns:
        Dictionary with keys:

        * ``model``           — metrics dict (rmse, mae, r2, mape) for the LSTM
        * ``naive``           — same metrics for the naive baseline
        * ``improvement_pct`` — percent RMSE reduction of model vs naive
        * ``y_true_real``     — inverse-transformed targets, ``(N, 1)``
        * ``y_pred_real``     — inverse-transformed model predictions
        * ``naive_real``      — inverse-transformed naive predictions
    """
    y_pred_scaled = model.predict(X, verbose=0)
    y_pred_real = scalers.inverse_transform_y(y_pred_scaled)
    y_true_real = scalers.inverse_transform_y(y)

    naive_scaled = naive_baseline(X, input_cols, baseline_feature)
    naive_real = scalers.inverse_transform_y(naive_scaled)

    model_metrics = compute_metrics(y_true_real, y_pred_real)
    naive_metrics = compute_metrics(y_true_real, naive_real)
    improvement = (1 - model_metrics["rmse"] / naive_metrics["rmse"]) * 100

    return {
        "model": model_metrics,
        "naive": naive_metrics,
        "improvement_pct": float(improvement),
        "y_true_real": y_true_real,
        "y_pred_real": y_pred_real,
        "naive_real": naive_real,
    }


# ---------------------------------------------------------------------------
# Plots — pass save_path to write a PNG; otherwise just returns the Figure
# ---------------------------------------------------------------------------
def plot_loss_curve(
    history,
    save_path: Path | str | None = None,
) -> "Figure":
    """Plot training vs. validation MSE loss across epochs.

    Args:
        history: A Keras ``History`` object returned by ``model.fit``.
        save_path: If provided, write a PNG to this path at 200 DPI.

    Returns:
        The Matplotlib :class:`Figure`.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="Train Loss", color="#2E86AB")
    ax.plot(history.history["val_loss"], label="Validation Loss",
            color="#E84855", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_predictions_timeseries(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual Volatility",
    save_path: Path | str | None = None,
) -> "Figure":
    """Plot predicted and actual volatility as overlaid time series.

    Args:
        y_true: Ground-truth volatility values, shape ``(N,)`` or ``(N, 1)``.
        y_pred: Predicted values, same shape.
        title: Plot title. Defaults to ``"Predicted vs Actual Volatility"``.
        save_path: If provided, write a PNG to this path at 200 DPI.

    Returns:
        The Matplotlib :class:`Figure`.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true, label="Actual", color="#2E86AB", linewidth=1)
    ax.plot(y_pred, label="Predicted", color="#E84855", linewidth=1, alpha=0.8)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Volatility")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_actual_vs_predicted_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual",
    save_path: Path | str | None = None,
) -> "Figure":
    """Scatter ``y_pred`` against ``y_true`` with a 45-degree reference line.

    Points falling on the dashed line are perfect predictions; deviation
    from the line is the per-window error.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.
        title: Plot title. Defaults to ``"Predicted vs Actual"``.
        save_path: If provided, write a PNG to this path at 200 DPI.

    Returns:
        The Matplotlib :class:`Figure`.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=5, color="#2E86AB")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Volatility")
    ax.set_ylabel("Predicted Volatility")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_naive_vs_model_rmse(
    naive_rmse: float,
    model_rmse: float,
    save_path: Path | str | None = None,
) -> "Figure":
    """Bar chart comparing the naive baseline's RMSE to the model's.

    The title displays the percent RMSE reduction.

    Args:
        naive_rmse: RMSE of the naive baseline.
        model_rmse: RMSE of the trained model.
        save_path: If provided, write a PNG to this path at 200 DPI.

    Returns:
        The Matplotlib :class:`Figure`.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["Naive Baseline", "LSTM Model"],
        [naive_rmse, model_rmse],
        color=["#888888", "#2E86AB"],
        edgecolor="black",
    )
    for bar, val in zip(bars, [naive_rmse, model_rmse]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.4f}", ha="center", va="bottom",
        )
    improvement = (1 - model_rmse / naive_rmse) * 100
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title(f"Model beats naive baseline by {improvement:.1f}%")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
