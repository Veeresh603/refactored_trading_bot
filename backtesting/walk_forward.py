"""
Walk-Forward Portfolio with Dynamic Allocation
----------------------------------------------
Supports:
- equal allocation
- custom weights
- sharpe-based allocation
- RL-based allocation
"""

import fastbt
import pandas as pd
import numpy as np
from backtesting.metrics import sharpe_ratio


class WalkForwardPortfolio:
    def __init__(self, strategies, allocation="equal", train_window=252, test_window=63, initial_balance=100000, rl_allocator=None):
        """
        strategies: list of strategy objects
        allocation: "equal", "sharpe", dict, or "rl"
        rl_allocator: RL agent (only used if allocation="rl")
        """
        self.strategies = strategies
        self.train_window = train_window
        self.test_window = test_window
        self.initial_balance = initial_balance
        self.allocation = allocation
        self.rl_allocator = rl_allocator

    def run(self, data: pd.DataFrame):
        n = len(data)
        portfolio_results = []
        portfolio_equity_curves = []

        for start in range(0, n - (self.train_window + self.test_window), self.test_window):
            train_data = data.iloc[start:start+self.train_window].copy()
            test_data = data.iloc[start+self.train_window:start+self.train_window+self.test_window].copy()

            strategy_balances = {}
            strategy_equities = {}
            strategy_returns = {}

            for strat in self.strategies:
                strat_name = strat.__class__.__name__

                strat.fit(train_data)
                strat_test = strat.generate_signals(test_data.copy())

                signal_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
                signals = [signal_map.get(s, 0) for s in strat_test["signal"].tolist()]

                trades = fastbt.backtest(strat_test["close"].tolist(), signals)

                pnl = sum([t.pnl for t in trades])
                final_balance = self.initial_balance + pnl
                strategy_balances[strat_name] = final_balance

                # Equity curve
                equity_curve = [self.initial_balance]
                for t in trades:
                    equity_curve.append(equity_curve[-1] + t.pnl)
                strategy_equities[strat_name] = pd.Series(equity_curve)
                strategy_returns[strat_name] = pd.Series(np.diff(equity_curve) / equity_curve[:-1])

            # -------------------------------
            # Dynamic Allocation
            # -------------------------------
            if self.allocation == "equal":
                weight = 1.0 / len(self.strategies)
                final_balance = sum(b * weight for b in strategy_balances.values())

            elif isinstance(self.allocation, dict):
                final_balance = sum(
                    strategy_balances[name] * self.allocation.get(name, 0)
                    for name in strategy_balances
                )

            elif self.allocation == "sharpe":
                sharpes = {name: sharpe_ratio(returns.dropna()) 
                           for name, returns in strategy_returns.items()}
                total = sum(abs(v) for v in sharpes.values())
                weights = {name: abs(v)/total for name, v in sharpes.items()}
                final_balance = sum(strategy_balances[name] * weights[name] for name in strategy_balances)

            elif self.allocation == "rl" and self.rl_allocator is not None:
                weights = self.rl_allocator.allocate(strategy_returns, online_update=True, update_steps=2000)
                final_balance = sum(strategy_balances[name] * weights[name] for name in strategy_balances)

            else:
                raise ValueError("Unsupported allocation method")

            portfolio_results.append(final_balance)

            # Combine equity curves (weighted sum)
            combined_equity = sum(strategy_equities.values()) / len(strategy_equities)
            portfolio_equity_curves.append(combined_equity)

        return portfolio_results, portfolio_equity_curves
