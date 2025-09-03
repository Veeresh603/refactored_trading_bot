"""
Base Strategy
-------------
- Defines common interface for all strategies
- Each strategy must implement generate_signals()
"""

import pandas as pd


class BaseStrategy:
    def __init__(self, name: str, asset: str = "NIFTY"):
        self.name = name
        self.asset = asset

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for given OHLCV data.
        Must return a DataFrame with at least ["time", "signal"].
        Signal: {1=Buy Call, -1=Buy Put, 0=Hold}
        """
        raise NotImplementedError("Strategy must implement generate_signals()")
