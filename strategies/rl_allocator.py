import os
import random
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
        self.strikes = strikes
        self.expiries = expiries

        # Env used for inference
        self.env = RLAllocatorEnv(asset_strategies, strikes, expiries)

        # Check if model exists
        model_file = model_path + ".zip"
        if not os.path.exists(model_file):
            print(f"⚠️ RL model not found at {model_file}, using random allocator instead")
            self.model = None
            self.obs = None
        else:
            self.model = PPO.load(model_path)
            self.obs = self.env.reset()

    def choose_action(self):
        """Predict next action = (strategy, strike_offset, expiry).
        Falls back to random if no model is available.
        """
        if self.model is None:
            # fallback: pick random strategy/strike/expiry
            strategy = random.choice(self.asset_strategies)
            strike_offset = random.choice(self.strikes)
            expiry = random.choice(self.expiries)
            return {
                "strategy": strategy,
                "strike_offset": strike_offset,
                "expiry": expiry
            }

        # RL agent prediction
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
