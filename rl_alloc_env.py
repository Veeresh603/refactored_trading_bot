import gymnasium as gym
import numpy as np

class PortfolioAllocEnv(gym.Env):
    """
    RL environment for dynamic portfolio allocation.
    - Observation: recent strategy returns
    - Action: weights across strategies
    - Reward: portfolio return (Sharpe-adjusted)
    """
    def __init__(self, strategy_returns, window=10):
        super(PortfolioAllocEnv, self).__init__()
        self.strategy_returns = strategy_returns  # dict {strat_name: pd.Series}
        self.strategies = list(strategy_returns.keys())
        self.n_strats = len(self.strategies)
        self.window = window

        # Action space: portfolio weights across strategies
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.n_strats,), dtype=np.float32)

        # Observation space: recent returns per strategy
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_strats * self.window,),
            dtype=np.float32
        )

        self.current_step = self.window

    def reset(self):
        self.current_step = self.window
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for strat in self.strategies:
            windowed = self.strategy_returns[strat].iloc[self.current_step - self.window:self.current_step].values
            obs.extend(windowed)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        # Normalize weights
        weights = np.clip(action, 0, 1)
        weights /= (weights.sum() + 1e-6)

        # Portfolio return = weighted sum of strategy returns
        strat_step_returns = [self.strategy_returns[s].iloc[self.current_step] for s in self.strategies]
        portfolio_return = np.dot(weights, strat_step_returns)

        # Reward = Sharpe-like adjustment (favor stable returns)
        reward = portfolio_return / (np.std(strat_step_returns) + 1e-6)

        self.current_step += 1
        done = self.current_step >= len(self.strategy_returns[self.strategies[0]]) - 1
        obs = self._get_obs()

        return obs, reward, done, {"weights": weights}
