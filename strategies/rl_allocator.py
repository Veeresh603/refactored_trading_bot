import os
import numpy as np
from stable_baselines3 import PPO
from strategies.rl_allocator_env import RLAllocatorEnv

class RLAllocator:
    def __init__(self, asset_strategies, multi_asset_returns, greek_exposures,
                 model_path="models/best_allocator_strike_expiry",
                 strikes=[-200, 0, 200], expiries=["weekly", "monthly"],
                 window_size=30):
        self.asset_strategies = asset_strategies
        self.multi_asset_returns = multi_asset_returns
        self.greek_exposures = greek_exposures
        self.window_size = window_size

        # Env used for inference
        self.env = RLAllocatorEnv(asset_strategies, strikes, expiries)

        if not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(f"❌ RL Allocator model not found at {model_path}")

        self.model = PPO.load(model_path)
        self.obs = self.env.reset()

    def choose_action(self):
        """Predict next action = (strategy, strike_offset, expiry)"""
        action, _ = self.model.predict(self.obs, deterministic=True)
        strategy_idx, strike_idx, expiry_idx = action

        strategy = self.asset_strategies[strategy_idx]
        strike_offset = self.env.strikes[strike_idx]
        expiry = self.env.expiries[expiry_idx]

        return {
            "strategy": strategy,
            "strike_offset": strike_offset,
            "expiry": expiry
        }
