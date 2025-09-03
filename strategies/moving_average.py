import pandas as pd
from strategies.base_strategy import BaseStrategy


class MovingAverageStrategy(BaseStrategy):
    def __init__(self, short_window=20, long_window=50, asset="NIFTY"):
        super().__init__(name="SMA", asset=asset)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["sma_short"] = df["close"].rolling(self.short_window).mean()
        df["sma_long"] = df["close"].rolling(self.long_window).mean()
        df["signal"] = 0
        df.loc[df["sma_short"] > df["sma_long"], "signal"] = 1
        df.loc[df["sma_short"] < df["sma_long"], "signal"] = -1
        return df[["time", "signal"]]
