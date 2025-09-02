"""
Backtesting Engine (C++ Accelerated)
------------------------------------
Uses fastbt (C++ extension) for lightning-fast backtesting
"""

import fastbt
import pandas as pd


class Backtester:
    def __init__(self, strategy, initial_balance=100000):
        self.strategy = strategy
        self.initial_balance = initial_balance

    def run(self, df: pd.DataFrame):
        """
        df: DataFrame with OHLCV
        strategy: must implement generate_signals(df)
        """
        # Generate signals from strategy
        df = self.strategy.generate_signals(df)

        # Convert signals to numeric: BUY=1, SELL=-1, HOLD=0
        signal_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
        signals = [signal_map.get(s, 0) for s in df["signal"].tolist()]

        # Run backtest in C++
        trades = fastbt.backtest(df["close"].tolist(), signals)

        # Convert trades to DataFrame
        trades_df = pd.DataFrame([t.__dict__ for t in trades])

        # Compute equity curve
        equity_curve = [self.initial_balance]
        for t in trades:
            equity_curve.append(equity_curve[-1] + t.pnl)

        return trades_df, pd.Series(equity_curve, name="equity")
