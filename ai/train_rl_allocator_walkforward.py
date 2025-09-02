"""
Walk-Forward Training for PPO Allocator
---------------------------------------
- Retrains PPO Allocator every week
- Uses rolling lookback window of strategy returns
- Evaluates Sharpe, Win-rate, Drawdown
- Saves only accepted models
"""

import os
import pandas as pd
from datetime import timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from strategies.rl_allocator_env import RLAllocatorEnv
from strategies.rl_eval import evaluate_model
from strategies.rl_report import generate_report


def train_allocator_walkforward(train_returns,
                                model_dir="models/allocator_walkforward",
                                report_dir="reports/allocator_walkforward",
                                lookback_days=365,
                                retrain_every_days=7,
                                total_timesteps=100000,
                                min_sharpe=0.5,
                                min_winrate=0.55,
                                max_drawdown=-0.2):

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # Align strategy returns into DataFrame
    df = pd.DataFrame(train_returns).dropna()

    # Ensure datetime index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    start_date = df.index.min() + timedelta(days=lookback_days)
    end_date = df.index.max()
    retrain_dates = pd.date_range(start=start_date, end=end_date, freq=f"{retrain_every_days}D")

    for d in retrain_dates:
        train_start = d - timedelta(days=lookback_days)
        train_data = df.loc[train_start:d]

        if len(train_data) < lookback_days // 2:
            continue

        print(f"📅 Retraining Allocator on {train_start.date()} → {d.date()} ({len(train_data)} rows)")

        env = DummyVecEnv([lambda: RLAllocatorEnv(train_data, window_size=30)])

        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./ppo_allocator_logs")
        model.learn(total_timesteps=total_timesteps)

        model_path = os.path.join(model_dir, f"ppo_allocator_{d.date()}")
        model.save(model_path + ".zip")

        # ✅ Evaluate
        metrics = evaluate_model(model_path + ".zip", train_data, window_size=30)
        print(f"📊 Allocator Metrics: {metrics}")

        # Generate report
        report_path = os.path.join(report_dir, f"report_allocator_{d.date()}.pdf")
        balances = train_data.sum(axis=1).cumsum().values  # proxy equity
        rewards = train_data.mean(axis=1).values           # proxy rewards
        generate_report(metrics, balances, rewards, report_path)

        # Accept / reject
        if (metrics["sharpe"] >= min_sharpe and
            metrics["win_rate"] >= min_winrate and
            metrics["max_dd"] >= max_drawdown):

            print(f"✅ Allocator Model accepted → {model_path}.zip")
        else:
            print(f"❌ Allocator Model rejected (Sharpe {metrics['sharpe']:.2f}, "
                  f"WinRate {metrics['win_rate']:.2f}, Drawdown {metrics['max_dd']:.2f})")
            os.remove(model_path + ".zip")
