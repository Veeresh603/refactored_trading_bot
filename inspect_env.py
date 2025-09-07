# inspect_env.py
from ai.train_rl import SimpleSyntheticEnv
env = SimpleSyntheticEnv(window=3, episode_length=10, seed=42, fill_delay_steps=1)
print("dir(env):", sorted([n for n in dir(env) if not n.startswith("_")]))
print("has attributes? idx:", hasattr(env, "idx"), "prices:", hasattr(env, "prices"),
      "position:", hasattr(env, "position"), "cash:", hasattr(env, "cash"))
# Show reprs for existing attrs
for name in ("idx", "prices", "position", "cash", "pending_orders"):
    try:
        print(name, "=", getattr(env, name))
    except Exception as e:
        print(name, "=>", repr(e))
