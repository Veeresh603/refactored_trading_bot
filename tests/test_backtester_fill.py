# tests/test_backtester_fill.py
"""
Unit tests for backtesting/backtest.py
Ensures fill-delay, slippage, and commission semantics work correctly.
"""

import numpy as np
import pandas as pd
from backtesting.backtest import Backtester


def make_price_df(n=10, start=100.0, step=1.0):
    """
    Build a simple synthetic price series.
    """
    prices = np.linspace(start, start + step * (n - 1), n)
    df = pd.DataFrame({"close": prices})
    df["time"] = pd.date_range("2025-01-01", periods=n, freq="1min")
    return df


def test_instant_fill_no_slippage_no_commission():
    df = make_price_df(5, start=100, step=1)
    # Signals: hold, long, hold, short, hold
    signals = [0, 1, 0, -1, 0]
    bt = Backtester(strategy=None, initial_balance=1000, fill_delay_steps=0, slippage_pct=0.0, commission=0.0, position_size=1)
    equity, trades, metrics = bt.run(df, signals=signals)
    # Should have 2 trades (long then flat)
    assert len(trades) == 2
    assert metrics["total_commission"] == 0.0
    assert metrics["total_trades"] == 2


def test_delayed_fill_changes_fill_index():
    df = make_price_df(5, start=100, step=1)
    signals = [0, 1, 0, -1, 0]
    bt = Backtester(strategy=None, initial_balance=1000, fill_delay_steps=1, slippage_pct=0.0, commission=0.0, position_size=1)
    equity, trades, metrics = bt.run(df, signals=signals)
    # First fill should happen at index 2 instead of 1 (delayed by 1)
    assert trades.iloc[0]["fill_idx"] == trades.iloc[0]["enqueue_idx"] + 1


def test_slippage_applied_correctly():
    df = make_price_df(3, start=100, step=0)
    signals = [0, 1, -1]  # buy then sell
    bt = Backtester(strategy=None, initial_balance=1000, fill_delay_steps=0, slippage_pct=0.01, commission=0.0, position_size=1)
    _, trades, _ = bt.run(df, signals=signals)
    buy_fill = trades.iloc[0]
    sell_fill = trades.iloc[1]
    # Buy should be 1% more expensive, sell 1% cheaper
    assert buy_fill["fill_price"] > 100.0
    assert sell_fill["fill_price"] < 100.0


def test_commission_reduces_cash():
    df = make_price_df(3, start=100, step=0)
    signals = [0, 1, -1]
    bt = Backtester(strategy=None, initial_balance=1000, fill_delay_steps=0, slippage_pct=0.0, commission=5.0, position_size=1)
    equity, trades, metrics = bt.run(df, signals=signals)
    # Two trades, total commission = 10
    assert metrics["total_commission"] == 10.0
    assert len(trades) == 2
    # Final equity should be lower than initial balance due to commission
    assert equity.iloc[-1] < 1000


def test_metrics_contains_expected_keys():
    df = make_price_df(5)
    signals = [0, 1, -1, 0, 0]
    bt = Backtester(strategy=None, initial_balance=1000, fill_delay_steps=0, slippage_pct=0.0, commission=0.0, position_size=1)
    _, _, metrics = bt.run(df, signals=signals)
    for key in ["final_return", "total_trades", "total_commission", "max_drawdown", "sharpe"]:
        assert key in metrics
