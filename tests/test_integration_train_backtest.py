# tests/test_integration_train_backtest.py
import os
import json
import tempfile

import pandas as pd
import numpy as np

from ai.train_rl import TrainConfig, RLTrainer
from backtesting.backtest import Backtester


def make_small_df(n=20, start_price=100.0, step=1.0):
    times = pd.date_range("2025-01-01", periods=n, freq="T")
    prices = [start_price + i * step for i in range(n)]
    df = pd.DataFrame({"time": times, "close": prices, "volume": np.ones(n) * 100.0})
    return df


def test_train_then_backtest_integration(tmp_path):
    # Short training config
    ckpt_dir = str(tmp_path / "checkpoints")
    cfg = TrainConfig(
        episodes=1,
        steps_per_episode=10,
        checkpoint_dir=ckpt_dir,
        seed=42,
        # keep defaults for other flags
    )

    trainer = RLTrainer(cfg)
    # Run a tiny training session (should not raise)
    trainer.train()

    # Check for run metadata or checkpoint files created
    run_files = [p for p in os.listdir(ckpt_dir) if p.endswith(".json") or p.endswith(".ckpt")]
    assert len(run_files) >= 1, f"No checkpoint/metadata written to {ckpt_dir}"

    # Create a tiny price df and run the backtester
    df = make_small_df(n=20)
    bt = Backtester(strategy=None, initial_balance=1000.0, fill_delay_steps=0, slippage_pct=0.0, commission=0.0, position_size=1.0)
    equity, trades, metrics = bt.run(df, signals=[0] * len(df))  # hold-only signals

    # assert metrics keys present
    for key in ("final_equity", "final_return", "total_trades", "total_commission", "max_drawdown", "sharpe"):
        assert key in metrics
