import fastindicators as fi
import pandas as pd

class MovingAverageStrategy:
    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window

    def fit(self, data: pd.DataFrame):
        # SMA doesn't need training, but ML/RL models would train here
        pass

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        prices = data["close"].tolist()
        sma_short = fi.sma(prices, self.short_window)
        sma_long = fi.sma(prices, self.long_window)

        data["signal"] = "HOLD"
        for i in range(len(prices)):
            if i >= self.long_window:
                if sma_short[i] > sma_long[i]:
                    data.at[i, "signal"] = "BUY"
                elif sma_short[i] < sma_long[i]:
                    data.at[i, "signal"] = "SELL"
        return data
