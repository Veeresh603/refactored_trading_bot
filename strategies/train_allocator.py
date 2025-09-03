"""
Train RL Allocator
------------------
- Trains PPO agent to act as allocator
- Observations: returns + Greeks
- Actions: choose strategy, strike offset, expiry, signal
- Reward: risk-adjusted PnL
- Logs training progress to TensorBoard
- Supports resuming training from a checkpoint
"""

import os
import gym
import numpy as np
import logging
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from core import execution_cpp
from strategies.rl_allocator_callback import AllocatorMetricsCallback

logger = logging.getLogger("TrainAllocator")


class AllocatorEnv(gym.Env):
    """
    RL Environment for allocator.
    State: [PnL normalized, delta, gamma, vega]
    Action: [strategy_idx, strike_idx, expiry_idx, signal]
    Reward: risk-adjusted PnL
    """
    def __init__(self, asset_strategies, strikes, expiries, spot_series,
                 max_steps=500, max_drawdown_penalty=0.5, margin_penalty=0.1):
        super(AllocatorEnv, self).__init__()

        self.asset_strategies = asset_strategies
        self.strikes = strikes
        self.expiries = expiries
        self.spot_series = spot_series
        self.max_steps = min(max_steps, len(spot_series) - 1)

        # Spaces
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([len(asset_strategies), len(strikes), len(expiries), 3])

        # State
        self.current_step = 0
        self.prev_equity = 100000
        self.peak_equity = self.prev_equity
        self.max_drawdown_penalty = max_drawdown_penalty
        self.margin_penalty = margin_penalty

        execution_cpp.reset_engine(self.prev_equity)

    def reset(self):
        self.current_step = 0
        self.prev_equity = 100000
        self.peak_equity = self.prev_equity
        execution_cpp.reset_engine(self.prev_equity)
        return np.zeros(4, dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        spot = self.spot_series[self.current_step]

        # Decode action
        strategy_idx, strike_idx, expiry_idx, signal_idx = action
        signal_val = 1 if signal_idx == 2 else -1 if signal_idx == 1 else 0
        strike_offset = self.strikes[strike_idx]
        expiry_type = self.expiries[expiry_idx]

        # Execute trade if signal != flat
        if signal_val != 0:
            strike = round(spot / 50) * 50 + strike_offset
            execution_cpp.place_order(
                symbol=f"NIFTY{strike}{'CE' if signal_val == 1 else 'PE'}",
                qty=50,
                price=spot,
                strike=strike,
                sigma=0.2,
                is_call=(signal_val == 1),
                expiry_days=30 if expiry_type == "monthly" else 7,
            )

        # Portfolio update
        status = execution_cpp.account_status(spot)
        equity = status["balance"] + status["total"]
        margin_used = status.get("margin_used", 0)

        # --- Reward ---
        pnl_change = equity - self.prev_equity
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = self.peak_equity - equity

        reward = pnl_change
        reward -= self.max_drawdown_penalty * (drawdown / 1000.0)
        reward -= self.margin_penalty * (margin_used / 1000.0)

        # --- Update state ---
        self.prev_equity = equity
        delta, gamma, vega, theta = execution_cpp.portfolio_greeks(spot)
        obs = np.array([pnl_change / 1000.0, delta, gamma, vega], dtype=np.float32)

        done = self.current_step >= self.max_steps
        info = {"equity": equity, "drawdown": drawdown, "margin_used": margin_used}

        return obs, reward, done, info

    def render(self, mode="human"):
        print(f"Step {self.current_step} | Equity={self.prev_equity:.2f}")


def train_allocator(timesteps=100000,
                    save_path="models/best_allocator_strike_expiry",
                    log_dir="logs/rl_allocator",
                    data_path="data/historical.csv",
                    resume_from=None):
    """
    Train PPO Allocator with risk-adjusted rewards + checkpointing.

    Args:
        timesteps (int): training steps
        save_path (str): final save path for PPO model
        log_dir (str): TensorBoard log dir
        data_path (str): historical data path
        resume_from (str): path to checkpoint (if resuming training)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Load market data
    df = pd.read_csv(data_path, parse_dates=["time"])
    spot_series = df["close"].values

    asset_strategies = ["SMA", "RSI", "MeanReversion"]
    strikes = [-100, 0, 100]
    expiries = ["weekly", "monthly"]

    env = AllocatorEnv(asset_strategies, strikes, expiries, spot_series)
    check_env(env, warn=True)

    logger.info(f"🚀 Training RLAllocator for {timesteps} timesteps (risk-adjusted)")
    if resume_from and os.path.exists(resume_from):
        logger.info(f"🔄 Loading checkpoint from {resume_from}")
        model = PPO.load(resume_from, env=env, verbose=1, tensorboard_log=log_dir)
    else:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    # Logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    # Attach callback
    callback = AllocatorMetricsCallback(save_path=os.path.join(save_path, "checkpoints"), verbose=1)

    # Train
    model.learn(total_timesteps=timesteps, callback=callback)

    # Save final model
    model.save(save_path)
    logger.info(f"✅ RLAllocator model saved at {save_path}")
    logger.info(f"📊 TensorBoard logs available at {log_dir}")
