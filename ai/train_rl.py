"""
RL Training Script
------------------
- Trains RL agent in trading environment
- Supports walk-forward training
- Logs metrics and saves best model
"""

import os
import logging
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from backtesting.metrics import compute_metrics

# Example: custom env
from envs.trading_env import TradingEnv
from ai.models.rl_agent import RLAgent


def train_rl(
    episodes=100,
    steps_per_episode=500,
    lr=1e-3,
    gamma=0.99,
    walkforward=False,
    save_path="models/rl_agent.pt",
):
    logger = logging.getLogger("RLTrainer")
    logger.setLevel(logging.INFO)

    env = TradingEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = RLAgent(obs_dim, act_dim, lr=lr, gamma=gamma)

    best_reward = -np.inf
    equity_curves = []

    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0
        ep_equity = []

        for step in range(steps_per_episode):
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)

            agent.remember(obs, action, reward, next_obs, done)
            agent.learn()

            obs = next_obs
            total_reward += reward
            ep_equity.append(info.get("equity", 0))

            if done:
                break

        logger.info(f"Episode {ep+1}/{episodes} → Reward={total_reward:.2f}")
        equity_curves.append({"time": ep, "equity": ep_equity[-1] if ep_equity else 0})

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.state_dict(), save_path)
            logger.info(f"💾 Best model saved @ {save_path}")

    # Compute metrics
    import pandas as pd
    eq_df = pd.DataFrame(equity_curves)
    metrics = compute_metrics(eq_df.rename(columns={"equity": "equity"}))

    logger.info(f"📊 Training Metrics: {metrics}")
    return metrics


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    os.makedirs("models", exist_ok=True)
    metrics = train_rl(episodes=50, steps_per_episode=200, walkforward=False)

    print("✅ RL Training Complete. Metrics:", metrics)


if __name__ == "__main__":
    main()
