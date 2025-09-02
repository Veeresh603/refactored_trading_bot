import gymnasium as gym
from gym import spaces
import numpy as np

class RLAllocatorEnv(gym.Env):
    def __init__(self, asset_strategies, strikes=[-200, 0, 200], expiries=["weekly", "monthly"]):
        super(RLAllocatorEnv, self).__init__()

        self.asset_strategies = asset_strategies
        self.strikes = strikes
        self.expiries = expiries

        # Action space: [strategy, strike offset, expiry type]
        self.action_space = spaces.MultiDiscrete([
            len(asset_strategies),
            len(strikes),
            len(expiries)
        ])

        # Observations = [returns, volatility, Greeks, etc.]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self.t = 0

    def reset(self):
        self.t = 0
        obs = np.zeros(self.observation_space.shape)
        return obs

    def step(self, action):
        strategy_idx, strike_idx, expiry_idx = action

        # Map actions
        strategy = self.asset_strategies[strategy_idx]
        strike_offset = self.strikes[strike_idx]
        expiry = self.expiries[expiry_idx]

        # TODO: replace with real backtest/market simulation
        reward = np.random.normal()  

        self.t += 1
        done = self.t > 100
        obs = np.random.randn(*self.observation_space.shape)  

        return obs, reward, done, {
            "strategy": strategy,
            "strike_offset": strike_offset,
            "expiry": expiry
        }
