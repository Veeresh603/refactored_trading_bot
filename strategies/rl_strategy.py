"""
RL Strategy
-----------
- Loads trained RL agent (Stable-Baselines3 PPO)
- Generates actions from market data
- Plug-and-play with StrategyEngine
"""

import os
import numpy as np
from stable_baselines3 import PPO


class RLStrategy:
    def __init__(self, model_dir="models/walkforward", window_size=30, asset="NIFTY"):
        self.model_dir = model_dir
        self.window_size = window_size
        self.asset = asset
        self.model = self._load_latest_model()

    def _load_latest_model(self):
        models = [f for f in os.listdir(self.model_dir) if f.endswith(".zip")]
        if not models:
            raise FileNotFoundError(f"No RL models found in {self.model_dir}")
        latest = sorted(models)[-1]
        return PPO.load(os.path.join(self.model_dir, latest))

    def generate_signals(self, obs: np.ndarray):
        """
        Generate signals from observation batch.

        Args:
            obs (np.ndarray): environment observations

        Returns:
            list[int]: signals (1=buy, -1=sell, 0=hold)
        """
        actions, _ = self.model.predict(obs)
        return actions.tolist()

    def evaluate(self, market_data):
        """
        Adapter for StrategyEngine (single timestep).
        """
        obs = market_data.get("obs")
        if obs is None:
            return {"asset": self.asset, "signal": 0}

        action, _ = self.model.predict(obs)
        signal = int(action)
        if signal == 1:
            return {"asset": self.asset, "signal": 1}
        elif signal == 2:
            return {"asset": self.asset, "signal": -1}
        return {"asset": self.asset, "signal": 0}
