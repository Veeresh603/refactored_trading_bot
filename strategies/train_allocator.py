from stable_baselines3 import PPO
from strategies.rl_allocator_env import RLAllocatorEnv

# Asset-strategy pairs
asset_strategies = [
    ("NIFTY", "SMA"),
    ("NIFTY", "RSI"),
    ("NIFTY", "RL")
]

# Create environment
env = RLAllocatorEnv(asset_strategies, strikes=[-200, 0, 200], expiries=["weekly", "monthly"])

# Train PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# Save best allocator model
model.save("models/best_allocator_strike_expiry")
print("✅ RL Allocator trained and saved at models/best_allocator_strike_expiry")
