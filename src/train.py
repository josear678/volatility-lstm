"""Scaler fitting (no leakage), training loop, artifact persistence, CLI.

Run from the project root::

    python -m src.train --epochs 500
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from . import config, data_loader, evaluate, features, model as model_mod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scaler bundle (fits on TRAIN portion only — no leakage)
# ---------------------------------------------------------------------------
class Scalers:
    """Bundles X and y :class:`MinMaxScaler` instances and tracks their columns.

    The class deliberately exposes no public method to fit on a full series:
    callers must pre-slice their DataFrames to the training portion before
    calling :meth:`fit`. This makes leakage from validation/test data into
    the scaler structurally impossible.

    Attributes:
        X: The feature scaler.
        y: The target scaler.
        numeric_cols: Column names the X scaler was fit on. Set after
            :meth:`fit`.
        target_col: Column name the y scaler was fit on. Set after
            :meth:`fit`.
    """

    def __init__(self) -> None:
        """Initialize empty scalers; column tracking is set during :meth:`fit`."""
        self.X: MinMaxScaler = MinMaxScaler()
        self.y: MinMaxScaler = MinMaxScaler()
        self.numeric_cols: list[str] | None = None
        self.target_col: str | None = None

    def fit(
        self,
        train_dfs: Iterable[pd.DataFrame],
        numeric_cols: list[str],
        target_col: str = config.TARGET_COL,
    ) -> "Scalers":
        """Fit X and y scalers on the concatenation of training DataFrames.

        Each DataFrame in ``train_dfs`` should already be sliced to the
        training window of that stock (e.g. first 80 % of rows). Validation
        and test rows must NOT appear here — that's the whole point of this
        method.

        Args:
            train_dfs: Iterable of DataFrames whose rows are training-only.
            numeric_cols: Column names to fit the X scaler on.
            target_col: Column name of the prediction target. Defaults to
                ``config.TARGET_COL``.

        Returns:
            ``self`` (for fluent chaining).
        """
        self.numeric_cols = list(numeric_cols)
        self.target_col = target_col
        X_concat = np.concatenate([df[numeric_cols].values for df in train_dfs])
        y_concat = np.concatenate(
            [df[[target_col]].values for df in train_dfs]
        ).reshape(-1, 1)
        self.X.fit(X_concat)
        self.y.fit(y_concat)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted scalers to a full DataFrame in place-on-copy.

        Safe to call on train / validation / test rows alike; only the
        scaler's *fit* must be train-only.

        Args:
            df: DataFrame containing all of ``self.numeric_cols`` and
                ``self.target_col``.

        Returns:
            A copy of ``df`` with those columns rescaled to ``[0, 1]``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.numeric_cols is None or self.target_col is None:
            raise RuntimeError("Scalers must be fit() before transform()")
        df = df.copy()
        df[self.numeric_cols] = self.X.transform(df[self.numeric_cols].values)
        df[self.target_col] = self.y.transform(df[[self.target_col]].values)
        return df

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        """Undo the y scaling — convert model outputs back to real units.

        Args:
            y: Scaled target values, any shape that flattens to 1-D.

        Returns:
            Real-units target values shaped ``(N, 1)``.
        """
        return self.y.inverse_transform(np.asarray(y).reshape(-1, 1))

    # ---- persistence ----
    def save(self, output_dir: Path | str) -> None:
        """Persist the fitted scalers and column-tracking metadata to disk.

        Writes ``X_scaler.pkl``, ``y_scaler.pkl``, and ``scaler_meta.json``
        into ``output_dir``.

        Args:
            output_dir: Destination directory. Created if it doesn't exist.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.X, output_dir / "X_scaler.pkl")
        joblib.dump(self.y, output_dir / "y_scaler.pkl")
        meta = {"numeric_cols": self.numeric_cols, "target_col": self.target_col}
        (output_dir / "scaler_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, model_dir: Path | str) -> "Scalers":
        """Load scalers previously written by :meth:`save`.

        Args:
            model_dir: Directory containing ``X_scaler.pkl``, ``y_scaler.pkl``,
                and ``scaler_meta.json``.

        Returns:
            A fully populated :class:`Scalers` instance.
        """
        model_dir = Path(model_dir)
        s = cls()
        s.X = joblib.load(model_dir / "X_scaler.pkl")
        s.y = joblib.load(model_dir / "y_scaler.pkl")
        meta = json.loads((model_dir / "scaler_meta.json").read_text())
        s.numeric_cols = meta["numeric_cols"]
        s.target_col = meta["target_col"]
        return s


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
def prepare_dataset(
    tickers: list[str] | None = None,
    *,
    use_vix: bool = True,
    window_size: int = config.WINDOW_SIZE,
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
    cache_dir: Path | str | None = None,
) -> dict:
    """Run the full data → ready-for-training pipeline.

    Steps, in order:

    1. Fetch each ticker's OHLC (and VIX, if ``use_vix``).
    2. Engineer features per stock.
    3. Slice each stock to its training portion and fit scalers (no leakage).
    4. Transform every stock's full history with the train-fitted scalers.
    5. Build sliding windows per stock.
    6. Chronologically split each stock's windows.
    7. Concatenate the per-split arrays across stocks.

    Args:
        tickers: List of tickers to use. Defaults to ``config.DEFAULT_TICKERS``.
        use_vix: If ``True`` (default), include the four VIX features.
        window_size: Sliding-window length. Defaults to ``config.WINDOW_SIZE``.
        train_frac: Fraction of each stock's history used for training.
            Defaults to ``config.TRAIN_FRAC``.
        val_frac: Fraction used for validation. Test fraction is the
            remainder. Defaults to ``config.VAL_FRAC``.
        cache_dir: Optional cache directory for yfinance downloads.

    Returns:
        Dictionary with keys:

        * ``X_train``, ``y_train``, ``X_val``, ``y_val``, ``X_test``,
          ``y_test`` — concatenated NumPy arrays
        * ``scalers`` — the fitted :class:`Scalers` instance
        * ``input_cols`` — list of feature column names
          (varies with ``use_vix``)
        * ``tickers`` — the list of tickers actually used
    """
    tickers = list(tickers) if tickers else list(config.DEFAULT_TICKERS)
    cache_dir = cache_dir if cache_dir is not None else config.DATA_CACHE_DIR

    logger.info("Fetching %d tickers (use_vix=%s)", len(tickers), use_vix)
    raw = data_loader.fetch_many(tickers, cache_dir=cache_dir)
    vix = data_loader.fetch_vix(cache_dir=cache_dir) if use_vix else None

    logger.info("Engineering features")
    engineered = {t: features.engineer_features(df, vix) for t, df in raw.items()}

    input_cols = config.INPUT_COLS if use_vix else config.STOCK_INPUT_COLS
    target_col = config.TARGET_COL

    # Slice train portion of each stock BEFORE fitting scalers (no leakage)
    train_dfs = [
        df.iloc[: int(np.floor(len(df) * train_frac))] for df in engineered.values()
    ]
    scalers = Scalers().fit(train_dfs, numeric_cols=input_cols, target_col=target_col)

    # Transform every stock's full history with the train-fitted scaler
    transformed = {t: scalers.transform(df) for t, df in engineered.items()}

    # Window + split each stock, then concatenate per-split across stocks
    splits: dict[str, list[np.ndarray]] = {
        k: [] for k in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")
    }
    for t, df in transformed.items():
        X, y = features.make_windows(df, input_cols, target_col, window_size)
        s = features.chronological_split(X, y, train_frac=train_frac, val_frac=val_frac)
        for k in splits:
            splits[k].append(s[k])

    return {
        **{k: np.concatenate(v) for k, v in splits.items()},
        "scalers": scalers,
        "input_cols": input_cols,
        "tickers": tickers,
    }


# ---------------------------------------------------------------------------
# Train + persist
# ---------------------------------------------------------------------------
def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = config.EPOCHS,
    patience: int = config.PATIENCE,
    learning_rate: float = config.LEARNING_RATE,
) -> tf.keras.callbacks.History:
    """Compile and fit ``model`` with EarlyStopping on ``val_loss``.

    Loss is MSE; the metric reported during training is RMSE. Best weights
    (lowest validation loss) are restored at the end of training.

    Args:
        model: An uncompiled Keras model (e.g. from :func:`src.model.build_lstm`).
        X_train: Training inputs.
        y_train: Training targets.
        X_val: Validation inputs (used by EarlyStopping).
        y_val: Validation targets.
        epochs: Hard upper bound on epochs; EarlyStopping usually exits earlier.
            Defaults to ``config.EPOCHS``.
        patience: EarlyStopping patience in epochs. Defaults to
            ``config.PATIENCE``.
        learning_rate: Adam learning rate. Defaults to ``config.LEARNING_RATE``.

    Returns:
        The Keras ``History`` object containing per-epoch loss / metric values.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    es = tf.keras.callbacks.EarlyStopping(
        patience=patience, restore_best_weights=True, monitor="val_loss"
    )
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[es],
    )


def save_artifacts(
    model: tf.keras.Model,
    scalers: Scalers,
    metadata: dict,
    output_dir: Path | str = config.MODELS_DIR,
) -> None:
    """Persist trained model, scalers, and run metadata.

    Writes:

    * ``volatility_lstm.keras``    — the Keras model
    * ``X_scaler.pkl``, ``y_scaler.pkl``, ``scaler_meta.json`` (via :meth:`Scalers.save`)
    * ``training_metadata.json``   — provenance record (tickers, hyperparams, dates)

    Args:
        model: Trained Keras model.
        scalers: Fitted :class:`Scalers` instance.
        metadata: JSON-serializable dict capturing the training run.
        output_dir: Destination directory. Defaults to ``config.MODELS_DIR``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "volatility_lstm.keras")
    scalers.save(output_dir)
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Saved artifacts to %s", output_dir)


def load_artifacts(
    model_dir: Path | str = config.MODELS_DIR,
) -> tuple[tf.keras.Model, Scalers, dict]:
    """Load a previously-saved (model, scalers, metadata) bundle.

    Args:
        model_dir: Directory previously written by :func:`save_artifacts`.
            Defaults to ``config.MODELS_DIR``.

    Returns:
        Tuple of ``(model, scalers, metadata)``. The scalers travel with the
        model so inference can never accidentally use mismatched scaling.
    """
    model_dir = Path(model_dir)
    model = tf.keras.models.load_model(model_dir / "volatility_lstm.keras")
    scalers = Scalers.load(model_dir)
    metadata = json.loads((model_dir / "training_metadata.json").read_text())
    return model, scalers, metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for ``python -m src.train``.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    p = argparse.ArgumentParser(description="Train the volatility LSTM.")
    p.add_argument("--tickers", nargs="+", default=config.DEFAULT_TICKERS,
                   help="Tickers to use for training.")
    p.add_argument("--no-vix", action="store_true",
                   help="Train without VIX features (uses %d inputs instead of %d)."
                        % (len(config.STOCK_INPUT_COLS), len(config.INPUT_COLS)))
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument("--output-dir", type=Path, default=config.MODELS_DIR)
    p.add_argument("--cache-dir", type=Path, default=config.DATA_CACHE_DIR)
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: prepare dataset, train, and save artifacts.

    Args:
        argv: Optional argument list (mainly for testing). When ``None``,
            arguments are read from :data:`sys.argv`.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Reproducibility (best effort — full determinism on GPU is not always possible)
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)

    use_vix = not args.no_vix
    ds = prepare_dataset(
        args.tickers, use_vix=use_vix, cache_dir=args.cache_dir,
    )
    model = model_mod.build_lstm(
        window_size=config.WINDOW_SIZE,
        num_features=len(ds["input_cols"]),
    )
    history = train_model(
        model,
        ds["X_train"], ds["y_train"],
        ds["X_val"], ds["y_val"],
        epochs=args.epochs, patience=args.patience,
    )

    epochs_run = len(history.history["loss"])
    early_stopped = epochs_run < args.epochs

    # Score the trained model on val/test so the metadata file is the
    # single source of truth for "what does this checkpoint achieve?"
    val_eval = evaluate.evaluate_on_split(
        model, ds["X_val"], ds["y_val"],
        scalers=ds["scalers"], input_cols=ds["input_cols"],
    )
    test_eval = evaluate.evaluate_on_split(
        model, ds["X_test"], ds["y_test"],
        scalers=ds["scalers"], input_cols=ds["input_cols"],
    )

    metadata = {
        "trained_on": datetime.now().isoformat(timespec="seconds"),
        "tickers": args.tickers,
        "use_vix": use_vix,
        "window_size": config.WINDOW_SIZE,
        "forecast_horizon": config.FORECAST_HORIZON,
        "input_cols": ds["input_cols"],
        "target_col": config.TARGET_COL,
        "lstm_units": config.LSTM_UNITS,
        "dense_units": config.DENSE_UNITS,
        "dropout": config.DROPOUT,
        "learning_rate": config.LEARNING_RATE,
        "epochs_requested": args.epochs,
        "epochs_run": epochs_run,
        "early_stopped": early_stopped,
        "patience": args.patience,
        "train_frac": config.TRAIN_FRAC,
        "val_frac": config.VAL_FRAC,
        "validation_metrics": {
            "model": val_eval["model"],
            "naive": val_eval["naive"],
            "improvement_pct_vs_naive": val_eval["improvement_pct"],
        },
        "test_metrics": {
            "model": test_eval["model"],
            "naive": test_eval["naive"],
            "improvement_pct_vs_naive": test_eval["improvement_pct"],
        },
    }
    save_artifacts(model, ds["scalers"], metadata, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
