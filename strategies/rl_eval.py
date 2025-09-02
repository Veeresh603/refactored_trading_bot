import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from strategies.rl_env import TradingEnv, add_indicators


def evaluate_model(model_path, df, window_size=30, n_eval_episodes=5):
    """
    Evaluate PPO model on historical test data.
    Returns Sharpe ratio, win-rate, max drawdown.
    """

    df = add_indicators(df)
    env = DummyVecEnv([lambda: TradingEnv(df, window_size=window_size)])

    model = PPO.load(model_path)

    rewards = []
    balances = []

    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        ep_rewards = []
        ep_balances = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_rewards.append(reward)
            if hasattr(env.envs[0], "balance"):
                ep_balances.append(env.envs[0].balance)

        rewards.extend(ep_rewards)
        balances.extend(ep_balances)

    rewards = np.array(rewards)
    balances = np.array(balances)

    # Sharpe ratio
    sharpe = np.mean(rewards) / (np.std(rewards) + 1e-9)

    # Win rate
    win_rate = np.sum(rewards > 0) / len(rewards)

    # Max drawdown
    cum_returns = pd.Series(balances)
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_dd = drawdown.min()

    return {"sharpe": sharpe, "win_rate": win_rate, "max_dd": max_dd}
