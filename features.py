"""
features.py — Ekstrakcja cech i etykietowanie anomalii

Trzy zestawy cech zgodnie z Eksperymentem 1:
  A — surowe zwroty dzienne
  B — rolling statistics (MA5, MA20, std5, std20)
  C — wskaźniki techniczne (RSI, Bollinger Bands)

Etykiety: anomalia gdy |z-score zwrotu| > z_thresh (domyślnie 3.0)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Etykietowanie
# ---------------------------------------------------------------------------

def _finalise(df: pd.DataFrame, cols: list):
    df = df.dropna()
    return df[cols].values, df["label"].values


def add_labels(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Dodaje kolumny 'return' i 'label' do DataFrame.
    label=1 gdy |z-score zwrotu dziennego| > z_thresh.
    """
    df = df.copy()
    ret = df["Close"].pct_change()
    z = (ret - ret.mean()) / ret.std()
    df["return"] = ret
    df["label"] = (np.abs(z) > z_thresh).astype(int)
    return df.dropna()


# ---------------------------------------------------------------------------
# Wskaźniki techniczne (helpers)
# ---------------------------------------------------------------------------

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _bollinger(close: pd.Series, period: int = 20):
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    width = (upper - lower) / ma
    pos = (close - lower) / (upper - lower)   # 0=at lower band, 1=at upper band
    return width, pos


# ---------------------------------------------------------------------------
# Zestawy cech
# ---------------------------------------------------------------------------

def feature_set_a(df: pd.DataFrame, z_thresh: float = 3.0):
    """Zestaw A: surowy zwrot dzienny (1 cecha)."""
    df = add_labels(df, z_thresh)
    return _finalise(df, ["return"])


def feature_set_b(df: pd.DataFrame, z_thresh: float = 3.0):
    """Zestaw B: rolling statistics — MA5, MA20, std5, std20 + zwrot (5 cech)."""
    df = add_labels(df, z_thresh)
    close = df["Close"]
    df["ma5"]  = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["std5"] = close.rolling(5).std()
    df["std20"] = close.rolling(20).std()
    return _finalise(df, ["return", "ma5", "ma20", "std5", "std20"])


def feature_set_c(df: pd.DataFrame, z_thresh: float = 3.0):
    """Zestaw C: RSI(14) + Bollinger Bands (20) + zwrot (4 cechy)."""
    df = add_labels(df, z_thresh)
    close = df["Close"]
    df["rsi"] = _rsi(close)
    df["bb_width"], df["bb_pos"] = _bollinger(close)
    return _finalise(df, ["return", "rsi", "bb_width", "bb_pos"])


def feature_set_all(df: pd.DataFrame, z_thresh: float = 3.0):
    """Wszystkie cechy łącznie (używane w Eksperymencie 2)."""
    df = add_labels(df, z_thresh)
    close = df["Close"]
    df["ma5"]   = close.rolling(5).mean()
    df["ma20"]  = close.rolling(20).mean()
    df["std5"]  = close.rolling(5).std()
    df["std20"] = close.rolling(20).std()
    df["rsi"] = _rsi(close)
    df["bb_width"], df["bb_pos"] = _bollinger(close)
    cols = ["return", "ma5", "ma20", "std5", "std20", "rsi", "bb_width", "bb_pos"]
    return _finalise(df, cols)


if __name__ == "__main__":
    from load_data import load_ticker
    ticker = input("Enter ticker (AAPL): ")
    df = load_ticker(ticker)
    for name, fn in [("A", feature_set_a), ("B", feature_set_b), ("C", feature_set_c)]:
        X, y = fn(df)
        print(f"Zestaw {name}: X={X.shape}, anomalie={y.sum()} ({100*y.mean():.2f}%)")
