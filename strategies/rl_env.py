"""
RL Environment (Gym-Compatible)
-------------------------------
- Wraps OHLCV data for RL training
- Features: RSI, MACD, Moving Averages
- Actions: Buy, Sell, Hold
- Rewards: PnL-based
- Integrated with AdvancedRiskManager
"""

import gymnasium as gym
from gym import spaces
import numpy as np
import pandas as pd
from core.risk_manager import AdvancedRiskManager


def add_indicators(df):
    """Add RSI, MACD, and Moving Averages to dataframe."""
    df = df.copy()

    # RSI (14)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Moving Averages
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()

    df = df.fillna(method="bfill")
    return df


class RLTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df, initial_balance=100000, window_size=30):
        super(RLTradingEnv, self).__init__()

        self.df = add_indicators(df)
        self.prices = self.df["close"].values
        self.features = self.df.drop(columns=["time"]).values
        self.window_size = window_size
        self.initial_balance = initial_balance

        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: window of features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, self.features.shape[1]), dtype=np.float32
        )

        self.risk_manager = AdvancedRiskManager()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0
        self.current_step = self.window_size
        self.trades = []
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        price = self.prices[self.current_step]
        reward = 0
        done = False
        info = {}

        # --- Execute Action ---
        if action == 1:  # Buy
            if self.position <= 0:
                self.position = 1
        elif action == 2:  # Sell
            if self.position >= 0:
                self.position = -1

        # --- PnL ---
        next_price = self.prices[self.current_step + 1] if self.current_step + 1 < len(self.prices) else price
        pnl = (next_price - price) * self.position
        self.equity += pnl
        reward = pnl

        # --- Risk Check ---
        safe, reason = self.risk_manager.check_risk(self.equity, pnl, abs(self.position))
        if not safe:
            done = True
            info["risk_triggered"] = reason

        # --- Log Trade ---
        self.trades.append(
            {"step": self.current_step, "action": action, "price": price, "pnl": pnl, "equity": self.equity}
        )

        # --- Advance ---
        self.current_step += 1
        if self.current_step >= len(self.prices) - 1:
            done = True

        obs = self._get_observation()
        info["equity"] = self.equity
        return obs, reward, done, False, info

    def _get_observation(self):
        start = self.current_step - self.window_size
        obs = self.features[start:self.current_step]
        return obs.astype(np.float32)

    def render(self, mode="human"):
        print(f"Step={self.current_step}, Equity={self.equity:.2f}, Position={self.position}")
