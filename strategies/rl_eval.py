"""
RL Evaluator
------------
- Evaluates a trained RL model on historical data
- Computes metrics (Sharpe, Win-Rate, Max Drawdown, etc.)
- Uses shared metrics module
- Optional PDF/Telegram reporting
"""

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from strategies.rl_env import RLTradingEnv, add_indicators
from backtesting.metrics import compute_metrics
from strategies.rl_reports import generate_rl_report


def evaluate_model(model_path, df, window_size=30, n_eval_episodes=5, report=True, send_telegram=False):
    """
    Evaluate PPO model on historical test data.

    Args:
        model_path (str): path to PPO model .zip
        df (pd.DataFrame): historical OHLCV data
        window_size (int): observation window size
        n_eval_episodes (int): number of evaluation runs
        report (bool): generate PDF/HTML report
        send_telegram (bool): send report to Telegram

    Returns:
        dict: performance metrics
    """
    df = add_indicators(df)
    env = DummyVecEnv([lambda: RLTradingEnv(df, window_size=window_size)])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")

    model = PPO.load(model_path)

    balances = []
    rewards = []

    for ep in range(n_eval_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        equity = 100000

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            equity += reward
            balances.append(equity)
            rewards.append(ep_reward)

    equity_curve = pd.DataFrame({"equity": balances})
    metrics = compute_metrics(equity_curve)

    print("\n📊 RL Evaluation Metrics")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.4f}" if isinstance(v, (int, float)) else f"{k:20s}: {v}")

    if report:
        generate_rl_report(metrics, balances, rewards, report_path="results/reports/rl_eval_report.pdf", send_telegram=send_telegram)

    return metrics


def main():
    import pandas as pd
    df = pd.read_csv("data/historical.csv", parse_dates=["time"])
    metrics = evaluate_model("models/best_allocator_strike_expiry.zip", df, window_size=30, n_eval_episodes=3, report=True, send_telegram=True)
    print(metrics)


if __name__ == "__main__":
    main()
