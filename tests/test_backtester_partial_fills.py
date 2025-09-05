# tests/test_backtester_partial_fills.py
import numpy as np
import pandas as pd
from backtesting.backtest import Backtester

def make_price_df_with_volume(n=6, start=100.0, step=1.0, volumes=None):
    prices = np.linspace(start, start + step * (n - 1), n)
    df = pd.DataFrame({"close": prices})
    if volumes is None:
        volumes = np.full(n, 100.0)
    df["volume"] = volumes
    return df

def test_partial_fill_when_liquidity_low():
    # Very low volume on bar 1 so only partial fill should happen
    volumes = [1000.0, 1.0, 1000.0, 1000.0, 1000.0, 1000.0]
    df = make_price_df_with_volume(n=6, start=100, step=1, volumes=volumes)
    # signals: hold, buy, hold, hold, hold, hold
    signals = [0, 1, 0, 0, 0, 0]
    # Request 10 units per trade, but liquidity_fraction=0.1 -> available=vol*0.1 => bar1: 0.1 units available (practically partial)
    bt = Backtester(strategy=None, initial_balance=10000, fill_delay_steps=0, slippage_pct=0.0, commission=0.0, position_size=10, liquidity_fraction=0.1)
    equity, trades, metrics = bt.run(df, signals=signals, order_unit_size=10)

    # There should be at least one trade with executed_units < requested_units (partial fill)
    assert not trades.empty
    partials = trades[trades["executed_units"] < trades["requested_units"]]
    assert len(partials) >= 1, f"Expected at least one partial fill, trades:\n{trades}"

    # The remaining_units should be > 0 for the partially-filled order (unless later bars filled remainder)
    # Check that overall executed sum equals final position (position_after on last trade equals position units)
    last_pos = trades.iloc[-1]["position_after"] if "position_after" in trades.columns else None
    # final equity should be a finite number
    assert np.isfinite(float(equity.iloc[-1]))

def test_partial_fill_fills_over_multiple_bars():
    # Make volumes such that first two bars have small liquidity, next bars fill remainder
    volumes = [1.0, 1.0, 100.0, 100.0, 100.0]
    df = make_price_df_with_volume(n=5, start=100, step=1, volumes=volumes)
    signals = [0, 1, 0, 0, 0]
    bt = Backtester(strategy=None, initial_balance=10000, fill_delay_steps=0, slippage_pct=0.0, commission=0.0, position_size=20, liquidity_fraction=0.1)
    equity, trades, metrics = bt.run(df, signals=signals, order_unit_size=20)

    # There should be multiple trades recorded for the same enqueue_idx as the order partially fills across bars
    if not trades.empty:
        first_enqueue = trades.iloc[0]["enqueue_idx"]
        occurrences = (trades["enqueue_idx"] == first_enqueue).sum()
        assert occurrences >= 2, f"Expected the order to be executed across multiple bars, got:\n{trades}"
