import os
import numpy as np
from stable_baselines3 import PPO
from strategies.rl_env import add_indicators


class RLStrategy:
    def __init__(self, model_dir="models/walkforward", window_size=30):
        self.model_dir = model_dir
        self.window_size = window_size
        self.model = self._load_latest_model()

    def _load_latest_model(self):
        models = [f for f in os.listdir(self.model_dir) if f.endswith(".zip")]
        if not models:
            raise FileNotFoundError("No accepted PPO models found in walkforward directory.")
        latest_model = max(models, key=lambda x: os.path.getctime(os.path.join(self.model_dir, x)))
        print(f"📂 Loading latest accepted PPO model: {latest_model}")
        return PPO.load(os.path.join(self.model_dir, latest_model))

    def generate_signals(self, df):
        df = df.copy()
        df = add_indicators(df)
        df["signal"] = "HOLD"

        if len(df) < self.window_size:
            return df

        obs = df[["open", "high", "low", "close", "volume", "rsi", "macd", "signal", "atr"]].tail(self.window_size).values
        obs = np.expand_dims(obs, axis=0)

        action, _ = self.model.predict(obs, deterministic=True)

        if action == 1:
            df.iloc[-1, df.columns.get_loc("signal")] = "BUY"
        elif action == 2:
            df.iloc[-1, df.columns.get_loc("signal")] = "SELL"

        return df
