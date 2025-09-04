"""
RL Allocator
------------
- Reinforcement Learning policy allocator (PPO)
- Chooses strategy, strike offset, and expiry
- Returns standardized decision dict for StrategyEngine
"""

import logging
import numpy as np
from stable_baselines3 import PPO

logger = logging.getLogger("RLAllocator")


class RLAllocator:
    def __init__(self, asset_strategies, multi_asset_returns, greek_exposures,
                 model_path="models/best_allocator_strike_expiry",
                 strikes=[-100, 0, 100], expiries=["weekly", "monthly"],
                 window_size=30):
        """
        Args:
            asset_strategies (list): list of strategies (tuples or names)
            multi_asset_returns (dict): recent returns
            greek_exposures (dict): portfolio Greeks
            model_path (str): saved PPO model
            strikes (list): strike offsets to choose from
            expiries (list): expiry choices
            window_size (int): lookback window
        """
        self.asset_strategies = asset_strategies
        self.multi_asset_returns = multi_asset_returns
        self.greek_exposures = greek_exposures
        self.model_path = model_path
        self.strikes = strikes
        self.expiries = expiries
        self.window_size = window_size

        try:
            self.model = PPO.load(model_path)
            logger.info(f"🤖 RLAllocator model loaded from {model_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load PPO model: {e}")
            self.model = None

    def choose_action(self):
        """
        Pick action using PPO policy or fallback to random.

        Returns:
            dict with keys:
                - strategy (str)
                - signal (int: 1=Buy Call, -1=Buy Put, 0=Hold)
                - strike_offset (int)
                - expiry (str)
        """
        if self.model is None:
            # fallback to random if model not loaded
            strategy = np.random.choice(self.asset_strategies)
            strike_offset = int(np.random.choice(self.strikes))
            expiry = str(np.random.choice(self.expiries))
            signal = np.random.choice([1, -1, 0])
            return {"strategy": "RLAllocator", "signal": signal,
                    "strike_offset": strike_offset, "expiry": expiry}

        # --- Build observation (very simplified) ---
        obs = np.array([np.mean(list(self.multi_asset_returns.values())) or 0.0,
                        self.greek_exposures.get("delta", 0.0),
                        self.greek_exposures.get("gamma", 0.0),
                        self.greek_exposures.get("vega", 0.0)])

        action, _ = self.model.predict(obs, deterministic=True)

        # Decode action (assuming discrete space: strategy, strike, expiry, signal)
        strategy_idx = action[0] % len(self.asset_strategies)
        strike_idx = action[1] % len(self.strikes)
        expiry_idx = action[2] % len(self.expiries)
        signal_val = 1 if action[3] > 0 else -1 if action[3] < 0 else 0

        decision = {
            "strategy": "RLAllocator",
            "signal": signal_val,
            "strike_offset": self.strikes[strike_idx],
            "expiry": self.expiries[expiry_idx],
        }

        logger.info(f"RLAllocator decision: {decision}")
        return decision
