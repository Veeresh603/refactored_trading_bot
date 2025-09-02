import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from strategies.rl_env import TradingEnv, add_indicators


def train_rl_agent(data_file="data/nifty50.csv", model_file="models/ppo_rl_trader"):
    # Load historical data
    df = pd.read_csv(data_file)
    df = df[["open", "high", "low", "close", "volume"]]
    df = add_indicators(df)

    # Create environment
    env = DummyVecEnv([lambda: TradingEnv(df, window_size=30)])

    # Train PPO
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs")
    model.learn(total_timesteps=200000)

    # Save model
    model.save(model_file)
    print(f"✅ PPO RL agent trained and saved to {model_file}")


if __name__ == "__main__":
    train_rl_agent()
