"""volatility-lstm — LSTM-based forecasting of one-week-ahead realized volatility.

Top-level re-exports so common operations are one import away::

    from src import (
        load_artifacts,        # load shipped (model, scalers, metadata)
        evaluate_on_split,     # run model + naive baseline on a split
        prepare_dataset,       # full data → tensors pipeline
        engineer_features,     # OHLC + VIX → feature DataFrame
        build_lstm,            # construct the model architecture
    )
"""
from .train import (
    Scalers,
    load_artifacts,
    prepare_dataset,
    save_artifacts,
    train_model,
)
from .evaluate import (
    compute_metrics,
    evaluate_on_split,
    naive_baseline,
)
from .features import (
    chronological_split,
    engineer_features,
    make_windows,
)
from .model import build_lstm

__all__ = [
    "Scalers",
    "build_lstm",
    "chronological_split",
    "compute_metrics",
    "engineer_features",
    "evaluate_on_split",
    "load_artifacts",
    "make_windows",
    "naive_baseline",
    "prepare_dataset",
    "save_artifacts",
    "train_model",
]

__version__ = "0.1.0"
