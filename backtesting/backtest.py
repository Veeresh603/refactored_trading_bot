"""
Backtesting Engine
------------------
- Uses C++ backtester (backtester_cpp) if available
- Falls back to Python implementation if missing
- Supports option-style signals (Buy Call, Buy Put, Hold)
- Computes performance metrics (Sharpe, Sortino, MaxDD, etc.)
"""

import logging
import pandas as pd

try:
    import backtester_cpp
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

from backtesting.metrics import compute_metrics

logger = logging.getLogger("Backtester")


class Backtester:
    def __init__(self, strategy, initial_balance=100000, fee_perc=0.001):
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.fee_perc = fee_perc

    def run(self, df: pd.DataFrame, strike=None, sigma=0.2, expiry_days=30):
        """
        Run backtest on OHLCV data + strategy signals.

        Args:
            df (pd.DataFrame): must contain "time", "close"
            strike (float): option strike (defaults to ATM at start)
            sigma (float): implied volatility
            expiry_days (int): expiry horizon in trading days
        """
        if "close" not in df.columns:
            raise ValueError("Data must contain 'close' column")

        # Generate strategy signals
        signals_df = self.strategy.generate_signals(df)
        if "signal" not in signals_df.columns:
            raise ValueError("Strategy must return a DataFrame with 'signal' column")

        prices = df["close"].tolist()
        sigs = signals_df["signal"].tolist()

        # Default strike = ATM from first price
        if strike is None:
            strike = round(df["close"].iloc[0] / 50) * 50

        # --- Use C++ Engine ---
        if HAS_CPP:
            logger.info("⚡ Using C++ backtester (backtester_cpp)")
            result = backtester_cpp.backtest_options(
                spot_prices=prices,
                signals=sigs,
                strike=strike,
                sigma=sigma,
                expiry_days=expiry_days,
                initial_balance=self.initial_balance,
                fee_perc=self.fee_perc,
            )
            equity_curve = pd.Series(result["equity_curve"], index=df["time"])
            trades_pnl = result["pnl"]

        # --- Python Fallback ---
        else:
            logger.warning("🐌 Falling back to Python backtester (slower)")
            equity_curve = [self.initial_balance]
            balance = self.initial_balance
            position = 0
            entry_price = 0.0
            trades_pnl = []

            for i in range(len(prices)):
                price = prices[i]
                signal = sigs[i]
                opt_price_call = max(price - strike, 0)
                opt_price_put = max(strike - price, 0)

                if signal == 1 and position == 0:
                    position = 1
                    entry_price = opt_price_call
                elif signal == -1 and position == 0:
                    position = -1
                    entry_price = opt_price_put
                elif signal == -1 and position == 1:
                    pnl = (opt_price_put - entry_price) - (opt_price_put * self.fee_perc)
                    balance += pnl
                    trades_pnl.append(pnl)
                    position = 0
                elif signal == 1 and position == -1:
                    pnl = (opt_price_call - entry_price) - (opt_price_call * self.fee_perc)
                    balance += pnl
                    trades_pnl.append(pnl)
                    position = 0
                equity_curve.append(balance)

            equity_curve = pd.Series(equity_curve, index=df["time"])

        # --- Metrics ---
        metrics = compute_metrics(pd.DataFrame({"equity": equity_curve}))
        return equity_curve, trades_pnl, metrics
