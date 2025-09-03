"""
Backtesting Metrics
-------------------
- Computes performance metrics for backtests
- Includes returns, Sharpe, Sortino, Max Drawdown, Win Rate, etc.
"""

import numpy as np
import pandas as pd


def compute_metrics(equity_curve: pd.DataFrame, risk_free_rate=0.05):
    """
    Compute performance metrics from equity curve.

    Args:
        equity_curve (pd.DataFrame): Must contain ["time", "equity"]
        risk_free_rate (float): Annual risk-free rate (default 5%)

    Returns:
        dict: performance metrics
    """
    if equity_curve.empty:
        return {}

    equity_curve = equity_curve.sort_values("time").reset_index(drop=True)
    equity = equity_curve["equity"].values
    returns = np.diff(equity) / equity[:-1]

    metrics = {}
    metrics["total_return"] = equity[-1] / equity[0] - 1
    metrics["cagr"] = _calc_cagr(equity_curve)
    metrics["sharpe"] = _calc_sharpe(returns, risk_free_rate)
    metrics["sortino"] = _calc_sortino(returns, risk_free_rate)
    metrics["max_drawdown"] = _calc_max_drawdown(equity)
    metrics["volatility"] = np.std(returns) * np.sqrt(252)
    metrics["win_rate"] = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
    metrics["avg_win"] = np.mean([r for r in returns if r > 0]) if np.any(returns > 0) else 0
    metrics["avg_loss"] = np.mean([r for r in returns if r < 0]) if np.any(returns < 0) else 0
    metrics["profit_factor"] = abs(metrics["avg_win"] / metrics["avg_loss"]) if metrics["avg_loss"] != 0 else np.inf

    return metrics


def _calc_cagr(equity_curve: pd.DataFrame):
    start_val = equity_curve["equity"].iloc[0]
    end_val = equity_curve["equity"].iloc[-1]
    days = (equity_curve["time"].iloc[-1] - equity_curve["time"].iloc[0]).days
    if days <= 0:
        return 0
    return (end_val / start_val) ** (252 / days) - 1


def _calc_sharpe(returns, risk_free_rate=0.05):
    if len(returns) == 0:
        return 0
    excess = returns - (risk_free_rate / 252)
    return np.mean(excess) / (np.std(returns) + 1e-9) * np.sqrt(252)


def _calc_sortino(returns, risk_free_rate=0.05):
    if len(returns) == 0:
        return 0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return np.inf
    excess = returns - (risk_free_rate / 252)
    return np.mean(excess) / (np.std(downside) + 1e-9) * np.sqrt(252)


def _calc_max_drawdown(equity):
    peak = equity[0]
    max_dd = 0
    for val in equity:
        peak = max(peak, val)
        dd = (peak - val) / peak
        max_dd = max(max_dd, dd)
    return max_dd
