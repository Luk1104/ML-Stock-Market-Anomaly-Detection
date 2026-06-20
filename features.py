"""
features.py - Ekstrakcja cech i etykietowanie anomalii

Trzy zestawy cech zgodnie z Eksperymentem 1:
  A - surowe zwroty dzienne (baseline)
  B - rolling statistics ceny + wolumen (MA5, MA20, std5, std20, vol_*)
  C - wskaźniki techniczne (RSI, Bollinger Bands) + wolumen

Etykiety: anomalia gdy |z-score zwrotu| > z_thresh ORAZ rolling z-score wolumenu > vol_thresh.
Wymóg obu warunków eliminuje fałszywe alarmy (np. rutynowe ogłoszenia wyników).
"""

import numpy as np
import pandas as pd
from config import Z_THRESH, VOL_THRESH


# ---------------------------------------------------------------------------
# Etykietowanie
# ---------------------------------------------------------------------------

def _finalise(df: pd.DataFrame, cols: list):
    df = df.dropna()
    return df[cols].values, df["label"].values


def add_labels(df: pd.DataFrame, z_thresh: float = Z_THRESH, vol_thresh: float = VOL_THRESH) -> pd.DataFrame:
    """
    Dodaje kolumny 'return' i 'label' do DataFrame.
    label=1 gdy |z-score zwrotu dziennego| > z_thresh ORAZ rolling z-score wolumenu > vol_thresh.
    Wolumen mierzony kroczącym z-score (okno 20 dni) - unika błędów wynikających z trendu wolumenu.
    """
    df = df.copy()
    ret = df["Close"].pct_change()
    z_ret = (ret - ret.mean()) / ret.std()
    df["return"] = ret

    vol      = df["Volume"].astype(float)
    vol_ma20 = vol.rolling(20).mean()
    vol_std20 = vol.rolling(20).std()
    vol_z    = (vol - vol_ma20) / vol_std20.replace(0, np.nan)

    # One-sided volume check: only high volume is suspicious, not low
    df["label"] = ((np.abs(z_ret) > z_thresh) & (vol_z > vol_thresh)).astype(int)
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


def _volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds volume-based columns to df in-place. Returns df."""
    vol = df["Volume"].astype(float)
    df["vol_ma5"]   = vol.rolling(5).mean()
    df["vol_ma20"]  = vol.rolling(20).mean()
    df["vol_std20"] = vol.rolling(20).std()
    df["vol_ratio"] = vol / df["vol_ma20"]   # ratio: 2.0 = twice the 20-day average
    return df


# ---------------------------------------------------------------------------
# Zestawy cech
# ---------------------------------------------------------------------------

def feature_set_a(df: pd.DataFrame, z_thresh: float = Z_THRESH, vol_thresh: float = VOL_THRESH):
    """Zestaw A: surowy zwrot dzienny (1 cecha). Baseline - celowo nie zawiera wolumenu."""
    df = add_labels(df, z_thresh, vol_thresh)
    return _finalise(df, ["return"])


def feature_set_b(df: pd.DataFrame, z_thresh: float = Z_THRESH, vol_thresh: float = VOL_THRESH):
    """Zestaw B: zwrot + rolling statistics ceny (MA, STD) + cechy wolumenu (VOLMA, VOLSTD, VOLRATIO) (9 cech)."""
    df = add_labels(df, z_thresh, vol_thresh)
    close = df["Close"]
    df["ma5"]  = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()
    df["std5"] = close.rolling(5).std()
    df["std20"] = close.rolling(20).std()
    df = _volume_features(df)
    return _finalise(df, ["return", "ma5", "ma20", "std5", "std20",
                           "vol_ma5", "vol_ma20", "vol_std20", "vol_ratio"])


def feature_set_c(df: pd.DataFrame, z_thresh: float = Z_THRESH, vol_thresh: float = VOL_THRESH):
    """Zestaw C: zwrot + RSI(14) + Bollinger Bands (20) + cechy wolumenu (6 cech)."""
    df = add_labels(df, z_thresh, vol_thresh)
    close = df["Close"]
    df["rsi"] = _rsi(close)
    df["bb_width"], df["bb_pos"] = _bollinger(close)
    df = _volume_features(df)
    return _finalise(df, ["return", "rsi", "bb_width", "bb_pos", "vol_ratio", "vol_std20"])


def feature_set_all(df: pd.DataFrame, z_thresh: float = Z_THRESH, vol_thresh: float = VOL_THRESH):
    """Wszystkie cechy łącznie (12 cech). Używane w Eksperymencie 2 i jako wejście dla Isolation Forest."""
    df = add_labels(df, z_thresh, vol_thresh)
    close = df["Close"]
    df["ma5"]   = close.rolling(5).mean()
    df["ma20"]  = close.rolling(20).mean()
    df["std5"]  = close.rolling(5).std()
    df["std20"] = close.rolling(20).std()
    df["rsi"]   = _rsi(close)
    df["bb_width"], df["bb_pos"] = _bollinger(close)
    df = _volume_features(df)
    cols = ["return", "ma5", "ma20", "std5", "std20",
            "rsi", "bb_width", "bb_pos",
            "vol_ma5", "vol_ma20", "vol_std20", "vol_ratio"]
    return _finalise(df, cols)


if __name__ == "__main__":
    from load_data import load_ticker
    ticker = input("Enter ticker (AAPL): ").strip() or "AAPL"
    df = load_ticker(ticker)
    for name, fn in [("A", feature_set_a), ("B", feature_set_b), ("C", feature_set_c), ("All", feature_set_all)]:
        X, y = fn(df)
        print(f"Zestaw {name}: X={X.shape}, anomalie={y.sum()} ({100*y.mean():.2f}%)")
