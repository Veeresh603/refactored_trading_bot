"""
Trading Environment (Gym-Compatible)
------------------------------------
- Step through OHLCV data
- Actions: Buy, Sell, Hold
- Rewards based on PnL
- Tracks equity, drawdown, risk
"""

import gym
import numpy as np
import pandas as pd
from gym import spaces
from core.risk_manager import AdvancedRiskManager


class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, data_path="data/historical.csv", initial_balance=100000, window_size=30):
        super(TradingEnv, self).__init__()

        self.data = pd.read_csv(data_path, parse_dates=["time"]).set_index("time")
        self.prices = self.data["close"].values
        self.window_size = window_size
        self.initial_balance = initial_balance

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: OHLCV window
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, self.data.shape[1]),
            dtype=np.float32,
        )

        self.risk_manager = AdvancedRiskManager()
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0  # +1 = long, -1 = short
        self.current_step = self.window_size
        self.trade_log = []
        return self._get_observation()

    def step(self, action):
        price = self.prices[self.current_step]

        reward = 0
        done = False
        info = {}

        # --- Execute action ---
        if action == 1:  # Buy
            if self.position <= 0:
                self.position = 1
        elif action == 2:  # Sell
            if self.position >= 0:
                self.position = -1
        else:  # Hold
            pass

        # --- PnL Calculation ---
        next_price = self.prices[self.current_step + 1] if self.current_step + 1 < len(self.prices) else price
        pnl = (next_price - price) * self.position
        self.equity += pnl
        reward = pnl

        # --- Risk Check ---
        safe, reason = self.risk_manager.check_risk(self.equity, pnl, abs(self.position))
        if not safe:
            done = True
            info["risk_triggered"] = reason

        # --- Logging ---
        self.trade_log.append(
            {"step": self.current_step, "action": action, "price": price, "pnl": pnl, "equity": self.equity}
        )

        # --- Advance Step ---
        self.current_step += 1
        if self.current_step >= len(self.prices) - 1:
            done = True

        obs = self._get_observation()
        info["equity"] = self.equity
        return obs, reward, done, info

    def _get_observation(self):
        start = self.current_step - self.window_size
        obs = self.data.iloc[start:self.current_step].values
        return obs.astype(np.float32)

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Equity: {self.equity:.2f}, Position: {self.position}")
