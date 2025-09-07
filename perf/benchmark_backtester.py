# perf/benchmark_backtester.py
import time
import numpy as np
import pandas as pd
from backtesting.backtest import Backtester

def make_large_df(n=200_000, start_price=100.0):
    rng = np.random.RandomState(42)
    steps = rng.normal(scale=1.0, size=n)
    prices = np.cumsum(np.concatenate([[start_price], steps]))[:n]
    df = pd.DataFrame({
        "close": prices,
        "volume": np.abs(rng.normal(loc=1000, scale=200, size=n))
    })
    return df

def simple_signals(df):
    # trivial alternating signals to force trades
    return np.tile([0, 1, 0, -1], int(np.ceil(len(df)/4)))[:len(df)]

def main():
    df = make_large_df(200_000)
    bt = Backtester(
        strategy=None,
        initial_balance=100000,
        fill_delay_steps=0,
        slippage_pct=0.0,
        commission=0.0,
        position_size=1
    )
    signals = simple_signals(df)
    t0 = time.time()
    eq, trades, metrics = bt.run(df, signals=signals, order_unit_size=1)
    t1 = time.time()
    print("✅ Benchmark done.")
    print("Rows:", len(df), "Trades:", len(trades))
    print("Time (s):", round(t1 - t0, 3))
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
