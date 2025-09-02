"""
Performance Metrics
-------------------
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Profit Factor
"""

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def sortino_ratio(returns: pd.Series, risk_free_rate=0.0):
    downside = returns[returns < 0]
    return np.sqrt(252) * (returns.mean() - risk_free_rate) / downside.std()


def max_drawdown(equity_curve: pd.Series):
    cum_max = equity_curve.cummax()
    dd = (equity_curve - cum_max) / cum_max
    return dd.min()


def profit_factor(trades: pd.DataFrame):
    gains = trades[trades["pnl"] > 0]["pnl"].sum()
    losses = -trades[trades["pnl"] < 0]["pnl"].sum()
    return gains / losses if losses > 0 else np.inf
# Metrics like Sharpe, Sortino, Max DD
