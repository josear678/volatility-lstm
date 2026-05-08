"""LSTM architecture for volatility forecasting.

This module is intentionally tiny — it only constructs the graph. Compilation
and training live in :mod:`src.train` so the architecture can be reused for
inference without dragging in optimizer state.
"""
from __future__ import annotations

import tensorflow as tf

from . import config


def build_lstm(
    window_size: int = config.WINDOW_SIZE,
    num_features: int = len(config.INPUT_COLS),
    lstm_units: int = config.LSTM_UNITS,
    dense_units: int = config.DENSE_UNITS,
    dropout: float = config.DROPOUT,
) -> tf.keras.Model:
    """Build (uncompiled) the volatility-forecasting LSTM.

    Architecture::

        Input(window_size, num_features)
          -> LSTM(lstm_units, tanh, return_sequences=True)
          -> LSTM(lstm_units, tanh)
          -> Dense(dense_units, relu)
          -> Dropout(dropout)
          -> Dense(1)

    The defaults give a deliberately small model (~14k parameters, ~210 KB
    when serialized). Two stacked LSTMs at 32 units empirically outperform
    a single 64-unit layer on this task; 16 dense units is enough capacity
    to mix the LSTM output without overfitting at this dataset size. The
    small footprint lets the model train on CPU in 10–15 minutes and ship
    inside the repo without external storage.

    Args:
        window_size: Number of timesteps in each input window. Defaults to
            ``config.WINDOW_SIZE``.
        num_features: Number of features per timestep. Defaults to the length
            of ``config.INPUT_COLS`` (currently 13: 9 stock + 4 VIX).
        lstm_units: Hidden size of each LSTM layer.
            Defaults to ``config.LSTM_UNITS``.
        dense_units: Units in the post-LSTM dense layer.
            Defaults to ``config.DENSE_UNITS``.
        dropout: Dropout probability applied after the dense layer.
            Defaults to ``config.DROPOUT``.

    Returns:
        An uncompiled :class:`tf.keras.Model`.
    """
    inputs = tf.keras.Input(shape=(window_size, num_features))
    x = tf.keras.layers.LSTM(lstm_units, activation="tanh", return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(lstm_units, activation="tanh", return_sequences=False)(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs, name="volatility_lstm")
