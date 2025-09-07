# tests/test_integration_train_eval_backtest.py
"""
Integration: train a tiny RL session, build signals by running trainer.agent over a small synthetic/historical df,
then run Backtester on those signals and assert basic metrics/trades exist.

This is intentionally small / deterministic and uses existing project components:
- ai.train_rl.RLTrainer + TrainConfig
- backtesting.backtest.Backtester
"""
import os
import json
import tempfile

import numpy as np
import pandas as pd

from ai.train_rl import RLTrainer, TrainConfig
from backtesting.backtest import Backtester

def make_price_df(n=30, start=100.0, step=0.5):
    times = pd.date_range("2025-01-01", periods=n, freq="T")
    prices = [start + i * step for i in range(n)]
    df = pd.DataFrame({"time": times.astype(str), "close": prices})
    return df

def test_train_then_backtest_integration(tmp_path):
    ckpt_dir = str(tmp_path / "checkpoints")
    cfg = TrainConfig(episodes=1, steps_per_episode=10, checkpoint_dir=ckpt_dir, seed=42)
    trainer = RLTrainer(cfg)
    # Run training (should not raise)
    trainer.train()

    # trainer.agent should exist (RandomAgent or model). Use it to generate signals on a small price df.
    df = make_price_df(n=30)
    # Build sliding-window observations consistent with SimpleSyntheticEnv.window if available
    window = getattr(trainer._make_env(), "window", 3)

    obs_buf = []
    signals = []
    # create last-window rolling observations for each bar; use agent.select_action
    for i in range(len(df)):
        start = max(0, i - window + 1)
        block = df["close"].tolist()[start : i + 1]
        if len(block) < window:
            block = [block[0]] * (window - len(block)) + block
        obs = np.array(block, dtype=float)
        act = trainer.agent.select_action(obs, deterministic=True)  # deterministic to get reproducible signals
        # map agent action to backtester signal space (must be -1,0,1)
        if act in (0, 1, -1):
            s = int(act)
        else:
            # fallback mapping if agent returns 0/1/2
            s = -1 if int(act) == 2 else (1 if int(act) == 1 else 0)
        signals.append(s)

    # run backtester
    bt = Backtester(strategy=None, initial_balance=10000.0, fill_delay_steps=0, slippage_pct=0.0, commission=0.0, position_size=1.0)
    equity, trades, metrics = bt.run(df.reset_index(drop=True), signals=signals)

    # Basic assertions: metrics has keys and equity length matches df length
    assert "final_equity" in metrics
    assert len(equity) == len(df)
    # trades is a dataframe (possibly empty): ensure metrics fields are present
    for k in ["final_equity", "final_return", "total_trades", "max_drawdown", "sharpe"]:
        assert k in metrics
