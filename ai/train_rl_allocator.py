from strategies.rl_allocator_env import RLAllocatorEnv
from strategies.rl_allocator_callback import AllocatorMetricsCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def train_rl_allocator(train_returns, greek_exposures, model_file="models/ppo_allocator_with_hedges"):
    env = DummyVecEnv([lambda: RLAllocatorEnv(train_returns, greek_exposures, window_size=30, dd_penalty_coeff=0.2)])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_allocator_logs")

    callback = AllocatorMetricsCallback(save_path="checkpoints", verbose=1)
    model.learn(total_timesteps=300000, callback=callback)

    # Save final model
    model.save(model_file)
    print(f"✅ Final PPO Allocator saved to {model_file}.zip")
