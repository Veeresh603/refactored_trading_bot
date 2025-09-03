import pandas as pd
from strategies.base_strategy import BaseStrategy


class RSIStrategy(BaseStrategy):
    def __init__(self, period=14, lower=30, upper=70, asset="NIFTY"):
        super().__init__(name="RSI", asset=asset)
        self.period = period
        self.lower = lower
        self.upper = upper

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
        rs = gain / loss.replace(0, 1)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["signal"] = 0
        df.loc[df["rsi"] < self.lower, "signal"] = 1
        df.loc[df["rsi"] > self.upper, "signal"] = -1
        return df[["time", "signal"]]
