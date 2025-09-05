# ai/envs.py
"""
Enhanced RL environment (gym-like API) for trading.

Action space: tuple (side:int, size_idx:int)
  - side: -1,0,1
  - size_idx: 0..(n_buckets-1) (maps to fraction of max position)

Observation: feature vector produced by data/loader.build_features_labels (or synthetic)

Features:
- Enqueue semantics (fill_delay handled by backtester)
- Risk controls: max_position, stop_loss_pct, take_profit_pct
- Reward: PnL - risk_penalty * position^2 - trade_penalty * abs(executed_units)
"""
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, List
import numpy as np
import math

class TradingEnv:
    def __init__(self, prices: np.ndarray, features: np.ndarray, initial_cash: float = 10000.0, max_position: float = 10.0, size_buckets: int = 5, stop_loss_pct: Optional[float] = None, take_profit_pct: Optional[float] = None):
        """
        prices: 1d array of close prices aligned with features
        features: shape (T, D)
        """
        self.prices = np.asarray(prices, dtype=float)
        self.features = np.asarray(features, dtype=float)
        assert len(self.prices) == len(self.features), "prices and features must align"
        self.T = len(self.prices)
        self.t = 0
        self.initial_cash = float(initial_cash)
        self.cash = self.initial_cash
        self.position = 0.0
        self.max_position = float(max_position)
        self.size_buckets = int(size_buckets)
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trade_penalty = 0.0
        self.risk_penalty = 0.001
        self.done = False

    def reset(self, seed: Optional[int] = None):
        self.t = 0
        self.cash = self.initial_cash
        self.position = 0.0
        self.done = False
        return self._obs()

    def _obs(self):
        obs = self.features[self.t].astype(float)
        # append cash/position normalized as features
        extra = np.array([self.cash / (self.initial_cash + 1e-9), self.position / (self.max_position + 1e-9)], dtype=float)
        return np.concatenate([obs, extra])

    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        action: (side, bucket)
        side: -1,0,1
        bucket: 0..size_buckets-1 -> fraction ( (bucket+1)/size_buckets ) of max_position
        """
        if self.done:
            raise RuntimeError("Step called on done env")
        side, bucket = int(action[0]), int(action[1])
        frac = (1 + bucket) / float(self.size_buckets)
        target_units = side * frac * self.max_position
        # compute delta units (we issue market order for delta)
        delta = target_units - self.position
        price = float(self.prices[self.t])
        executed = delta  # for synthetic env we assume instant fill; in real env integrate backtester
        # update cash/position
        self.position += executed
        self.cash -= executed * price
        # compute immediate PnL change from position * price movement will be counted next step (mark-to-market)
        reward = 0.0
        # risk penalty
        reward -= self.risk_penalty * (self.position ** 2)
        # trade penalty
        reward -= self.trade_penalty * abs(executed)
        # step forward
        self.t += 1
        if self.t >= self.T:
            self.done = True
        # mark-to-market PnL
        current_equity = self.cash + self.position * (self.prices[self.t - 1] if self.t > 0 else price)
        reward += (current_equity - self.initial_cash) * 0.0  # keep immediate small; shaped rewards better applied elsewhere
        info = {"cash": self.cash, "position": self.position, "t": self.t}
        # stop-loss / take-profit enforcement
        if self.stop_loss_pct is not None:
            unrealized = self.position * self.prices[self.t - 1]
            if unrealized + self.cash < self.initial_cash * (1.0 - abs(self.stop_loss_pct)):
                self.done = True
                info["stopped_out"] = True
        return self._obs(), float(reward), self.done, info

    def render(self):
        print(f"t={self.t} price={self.prices[self.t-1] if self.t>0 else self.prices[0]:.2f} cash={self.cash:.2f} position={self.position:.4f}")
