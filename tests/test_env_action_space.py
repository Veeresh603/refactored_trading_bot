# tests/test_env_action_space.py
import numpy as np
from ai.envs import TradingEnv

def make_simple_prices_features(n=10):
    prices = np.linspace(100.0, 110.0, n)
    # features: simple constant features
    features = np.vstack([np.linspace(0, 1, n) for _ in range(4)]).T
    return prices, features

def test_env_actions_and_limits():
    prices, features = make_simple_prices_features(20)
    env = TradingEnv(prices=prices, features=features, initial_cash=1000.0, max_position=5.0, size_buckets=4)
    obs = env.reset()
    # action: go long with largest bucket
    obs, reward, done, info = env.step((1, 3))
    assert env.position != 0.0
    # try reverse side to flatten
    obs, reward2, done2, info2 = env.step((0, 0))  # target flat
    # ensure position moves towards 0 or becomes 0 by the end
    assert abs(env.position) <= env.max_position + 1e-6

def test_stop_loss_triggers():
    prices, features = make_simple_prices_features(3)
    env = TradingEnv(prices=prices, features=features, initial_cash=1000.0, max_position=1.0, size_buckets=1, stop_loss_pct=0.01)
    env.reset()
    # Take large long then simulate bad move by stepping until stop triggers
    env.step((1, 0))
    # Force cash+unrealized below threshold by manipulating internal state
    env.cash = 0.0
    env.position = -100.0
    obs, r, done, info = env.step((0, 0))
    assert done or info.get("stopped_out", False)
