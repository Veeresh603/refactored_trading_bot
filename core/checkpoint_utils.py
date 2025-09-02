import numpy as np
from stable_baselines3 import PPO
from strategies.rl_allocator_env import RLAllocatorEnv


def quick_evaluate_checkpoint(model_path, train_returns, greek_exposures, window_size=30):
    """
    Quick evaluation of Allocator checkpoint
    - Returns Sharpe, Max Drawdown
    - Used for failover validation
    """
    try:
        model = PPO.load(model_path)
        env = RLAllocatorEnv(train_returns, greek_exposures, window_size=window_size)

        obs = env.reset()
        done = False
        rewards = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)

        sharpe = np.mean(rewards) / (np.std(rewards) + 1e-8)
        equity_curve = np.cumprod([1 + r for r in rewards])
        peak = np.maximum.accumulate(equity_curve)
        max_dd = ((equity_curve - peak) / peak).min()

        return {"sharpe": sharpe, "max_dd": max_dd}
    except Exception as e:
        return {"sharpe": -999, "max_dd": -999, "error": str(e)}
