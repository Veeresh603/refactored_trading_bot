"""
Feature Engineering
-------------------
- Generate indicators
- Create ML-ready feature sets
"""

import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to OHLCV dataframe
    """
    df["SMA_20"] = df["close"].rolling(20).mean()
    df["SMA_50"] = df["close"].rolling(50).mean()
    df["RSI"] = compute_rsi(df["close"], 14)
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI)
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
