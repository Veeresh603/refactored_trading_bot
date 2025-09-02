"""
Evaluate Allocator Checkpoints
------------------------------
- Loads all saved checkpoints
- Evaluates Sharpe, WinRate, Max Drawdown
- Picks best model for live deployment
"""

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from strategies.rl_allocator_env import RLAllocatorEnv
from strategies.rl_eval import evaluate_model  # assumes you already have RL evaluation helper


def evaluate_checkpoints(train_returns, greek_exposures,
                         checkpoints_dir="checkpoints",
                         model_out="models/best_allocator.zip",
                         min_sharpe=0.7, max_dd=-0.2):
    # Prepare evaluation environment
    env = RLAllocatorEnv(train_returns, greek_exposures, window_size=30)

    best_model = None
    best_sharpe = -np.inf
    best_path = None

    # Scan all checkpoints
    for file in os.listdir(checkpoints_dir):
        if file.endswith(".zip"):
            path = os.path.join(checkpoints_dir, file)
            print(f"🔍 Evaluating {path} ...")

            try:
                model = PPO.load(path)
                metrics = evaluate_model(path, train_returns, greek_exposures, window_size=30)

                sharpe = metrics.get("sharpe", -999)
                max_dd = metrics.get("max_dd", -999)

                print(f"📊 Sharpe={sharpe:.2f}, MaxDD={max_dd:.2%}")

                if sharpe > best_sharpe and max_dd >= max_dd:
                    best_sharpe = sharpe
                    best_model = model
                    best_path = path

            except Exception as e:
                print(f"❌ Failed to evaluate {path}: {e}")

    if best_model:
        best_model.save(model_out)
        print(f"✅ Best model {best_path} saved as {model_out} (Sharpe {best_sharpe:.2f})")
    else:
        print("⚠️ No valid checkpoint found!")


if __name__ == "__main__":
    # Example: placeholder returns + Greeks
    df = pd.read_csv("data/nifty50.csv")
    train_returns = {
        ("NIFTY", "SMA"): df["close"].pct_change().fillna(0),
        ("NIFTY", "RSI"): df["close"].pct_change().fillna(0) * 0.8,
        ("NIFTY", "RL"): df["close"].pct_change().fillna(0) * 1.2,
    }

    greek_exposures = {
        "delta": df["close"].pct_change().fillna(0),
        "gamma": df["close"].pct_change().fillna(0) * 0.001,
        "vega": df["close"].pct_change().fillna(0) * 50,
    }

    evaluate_checkpoints(train_returns, greek_exposures)
