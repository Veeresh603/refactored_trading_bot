import gymnasium as gym
from gym import spaces
import numpy as np
import pandas as pd


def add_indicators(df):
    """Add technical indicators to the OHLCV dataframe."""
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

    # ATR (14)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = high_low.to_frame().join(high_close).join(low_close).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # Drop NaNs
    df = df.dropna().reset_index(drop=True)
    return df


class TradingEnv(gym.Env):
    """
    Custom Trading Environment for RL (discrete actions)
    Actions: 0=HOLD, 1=BUY, 2=SELL
    Observation: last N candles (OHLCV + indicators)
    Reward: PnL changes
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, df, window_size=30, initial_balance=100000):
        super(TradingEnv, self).__init__()

        # Add indicators
        df = add_indicators(df)
        self.df = df.reset_index(drop=True)

        self.window_size = window_size
        self.initial_balance = initial_balance

        # Spaces
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, self.df.shape[1]), dtype=np.float32
        )

        # State
        self.balance = initial_balance
        self.position = 0  # +1 = long, -1 = short, 0 = flat
        self.entry_price = 0
        self.current_step = window_size

    def _get_observation(self):
        return self.df.iloc[self.current_step - self.window_size:self.current_step].values

    def step(self, action):
        price = self.df.iloc[self.current_step]["close"]
        reward = 0

        # HOLD
        if action == 0:
            pass

        # BUY
        elif action == 1:
            if self.position == 0:
                self.position = 1
                self.entry_price = price
            elif self.position == -1:  # closing short
                reward = self.entry_price - price
                self.balance += reward
                self.position = 0

        # SELL
        elif action == 2:
            if self.position == 0:
                self.position = -1
                self.entry_price = price
            elif self.position == 1:  # closing long
                reward = price - self.entry_price
                self.balance += reward
                self.position = 0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self._get_observation()

        return obs, reward, done, {}

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = self.window_size
        return self._get_observation()

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}")
