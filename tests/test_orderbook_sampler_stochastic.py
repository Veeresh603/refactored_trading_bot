# tests/test_orderbook_sampler_stochastic.py
import numpy as np
import pandas as pd
from backtesting.orderbook_sampler import OrderBookSampler

def make_df(n=5):
    return pd.DataFrame({
        "close": np.linspace(100, 100 + n - 1, n),
        "high": np.linspace(101, 101 + n - 1, n),
        "low": np.linspace(99, 99 + n - 1, n),
        "volume": np.array([10.0, 5.0, 0.0, 50.0, 1.0]),
    })

def test_available_liquidity_seed_deterministic():
    df = make_df()
    s1 = OrderBookSampler(df=df, liquidity_fraction=0.2, seed=42)
    s2 = OrderBookSampler(df=df, liquidity_fraction=0.2, seed=42)
    a1 = [s1.available_liquidity(i) for i in range(len(df))]
    a2 = [s2.available_liquidity(i) for i in range(len(df))]
    assert a1 == a2  # reproducible with same seed

def test_execute_respects_available():
    df = make_df()
    s = OrderBookSampler(df=df, liquidity_fraction=0.1, seed=7)
    avail = s.available_liquidity(0)
    res = s.execute(0, side=1, requested_units=avail * 2.0)
    assert res["executed"] <= avail + 1e-12
    assert "vwap" in res
