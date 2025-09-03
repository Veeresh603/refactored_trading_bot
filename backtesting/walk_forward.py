"""
Walk-Forward Backtesting
------------------------
- Splits data into rolling train/test windows
- Retrains models (if applicable) on train sets
- Tests on next window using Backtester
"""

import pandas as pd
import logging
from backtesting.backtest import Backtester


class WalkForwardTester:
    def __init__(self, broker=None, initial_balance=100000, risk_cfg=None, exec_cfg=None,
                 rl_allocator=None, options_optimizer=None, ensemble=None,
                 train_size=252, test_size=63):
        """
        Args:
            broker: broker interface (optional, not used in backtest)
            initial_balance: starting capital
            risk_cfg: dict of risk manager config
            exec_cfg: dict of execution config
            rl_allocator: RLAllocator instance
            options_optimizer: OptionsOptimizer instance
            ensemble: EnsembleEngine instance
            train_size: rolling training window (days)
            test_size: rolling test window (days)
        """
        self.logger = logging.getLogger("WalkForwardTester")
        self.initial_balance = initial_balance
        self.risk_cfg = risk_cfg
        self.exec_cfg = exec_cfg
        self.rl_allocator = rl_allocator
        self.options_optimizer = options_optimizer
        self.ensemble = ensemble
        self.train_size = train_size
        self.test_size = test_size

    def run(self, df: pd.DataFrame, asset="NIFTY", iv=0.2, trend="neutral"):
        """
        Run walk-forward backtest.

        Args:
            df (DataFrame): OHLCV data with datetime index
            asset (str): trading asset
            iv (float): default implied volatility
            trend (str): market assumption

        Returns:
            dict: aggregated results {equity_curve, trades, metrics}
        """
        results = []
        all_trades = []
        all_equity = []

        start = 0
        while start + self.train_size + self.test_size <= len(df):
            train_df = df.iloc[start:start + self.train_size]
            test_df = df.iloc[start + self.train_size:start + self.train_size + self.test_size]

            self.logger.info(f"🔄 Training on {train_df.index[0].date()} → {train_df.index[-1].date()}, "
                             f"testing on {test_df.index[0].date()} → {test_df.index[-1].date()}")

            # TODO: retrain ensemble/RL models here if needed
            # e.g. self.ensemble.retrain(train_df)

            backtester = Backtester(
                broker=None,
                initial_balance=self.initial_balance if not results else results[-1]["equity_curve"]["equity"].iloc[-1],
                risk_cfg=self.risk_cfg,
                exec_cfg=self.exec_cfg,
                rl_allocator=self.rl_allocator,
                options_optimizer=self.options_optimizer,
                ensemble=self.ensemble,
            )

            result = backtester.run(test_df, asset=asset, iv=iv, trend=trend)

            results.append(result)
            all_trades.append(result["trades"])
            all_equity.append(result["equity_curve"])

            start += self.test_size  # roll forward

        # Combine
        combined_trades = pd.concat(all_trades).reset_index(drop=True) if all_trades else pd.DataFrame()
        combined_equity = pd.concat(all_equity).reset_index(drop=True) if all_equity else pd.DataFrame()

        return {
            "equity_curve": combined_equity,
            "trades": combined_trades,
            "metrics": results[-1]["metrics"] if results else {},
        }
