import pandas as pd
from strategies.base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    def __init__(self, lookback=20, threshold=0.02, asset="NIFTY"):
        super().__init__(name="MeanReversion", asset=asset)
        self.lookback = lookback
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["rolling_mean"] = df["close"].rolling(self.lookback).mean()
        df["deviation"] = (df["close"] - df["rolling_mean"]) / df["rolling_mean"]
        df["signal"] = 0
        df.loc[df["deviation"] > self.threshold, "signal"] = -1
        df.loc[df["deviation"] < -self.threshold, "signal"] = 1
        return df[["time", "signal"]]
