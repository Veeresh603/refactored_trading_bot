# tests/test_env_partial_fill_integration.py
import numpy as np
from ai.train_rl import SimpleSyntheticEnv

def test_env_partial_fill_with_fallback_liquidity():
    env = SimpleSyntheticEnv(window=3, episode_length=10, seed=42, fill_delay_steps=0, liquidity_fraction=0.01, slippage_pct=0.0, commission=0.0)
    obs = env.reset()
    # low liquidity -> partial fills expected
    obs, r, done, info = env.step(1)  # enqueue buy
    # step again to allow fill if fill_delay_steps==0
    obs, r2, done2, info2 = env.step(0)
    # info2 should include fill info when partial fills occur
    assert isinstance(info2.get("fill"), bool)
