"""Project-wide configuration constants.

Edit values here rather than scattering hyperparameters across modules.
"""
from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------
DEFAULT_TICKERS: list[str] = [
    "TSLA", "NVDA", "AAPL", "MSFT", "GOOGL",
    "BRK-B", "JNJ", "KO", "CVX", "^GSPC",
]
HOLDOUT_TICKERS: list[str] = ["AMZN", "META", "LLY"]
VIX_TICKER: str = "^VIX"

# ---------------------------------------------------------------------------
# Windowing / forecast horizon
# ---------------------------------------------------------------------------
WINDOW_SIZE: int = 30
FORECAST_HORIZON: int = 5  # predict realized weekly volatility 5 trading days ahead

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
TRAIN_FRAC: float = 0.8
VAL_FRAC: float = 0.1  # implies TEST_FRAC = 0.1

# ---------------------------------------------------------------------------
# Feature columns the model consumes
# ---------------------------------------------------------------------------
STOCK_INPUT_COLS: list[str] = [
    "Volume", "return", "daily_range", "daily_volatility",
    "weekly_volatility_fast", "weekly_volatility_slow",
    "yearly_volatility", "MA_20", "MA_50",
]
VIX_INPUT_COLS: list[str] = [
    "vix_close", "vix_return", "vix_ma_20", "vix_rolling_std_20",
]
INPUT_COLS: list[str] = STOCK_INPUT_COLS + VIX_INPUT_COLS  # 13 features
TARGET_COL: str = "target_volatility"

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
LSTM_UNITS: int = 32
DENSE_UNITS: int = 16
DROPOUT: float = 0.3

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
LEARNING_RATE: float = 1e-3
EPOCHS: int = 500
PATIENCE: int = 15
BATCH_SIZE: int = 32  # Keras default; left explicit for documentation

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
