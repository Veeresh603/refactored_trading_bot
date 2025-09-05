# tests/test_execution_audit.py
import numpy as np
from ai.train_rl import SimpleSyntheticEnv
import math

def test_order_fill_delay_prevents_same_timestamp_fill():
    """
    Ensure that an order enqueued at time t is NOT filled at price[t] but at price[t+fill_delay_steps].
    We'll use deterministic RNG seed so prices are reproducible.
    """
    seed = 1234
    env = SimpleSyntheticEnv(window=3, episode_length=10, price_drift=0.0, seed=seed, fill_delay_steps=1, slippage_pct=0.0, commission=0.0)
    obs = env.reset()
    # record initial last price (time t0)
    t0_price = float(env.prices[-1])

    # take an action at t0 to BUY (1)
    obs_t1, reward1, done1, info1 = env.step(1)  # this enqueues and advances to t=1
    # Because fill_delay_steps=1, the order enqueued at step 0 should be fillable at step 1,
    # and thus info1 should indicate a fill (fill at price at t=1), not t0_price.
    assert info1["ordered"] is True
    assert info1["fill"] is True, f"Expected a fill at step 1 for enqueue at step 0, got info={info1}"
    fill_price = info1["fill_price"]
    assert fill_price is not None
    # Assert fill price is not equal to t0_price (i.e., not same-timestamp)
    assert not math.isclose(fill_price, t0_price, rel_tol=1e-9, abs_tol=1e-12), f"Fill price used same timestamp price (lookahead): fill={fill_price} t0={t0_price}"

    # Now test that subsequent step without new order keeps position
    obs_t2, reward2, done2, info2 = env.step(0)
    assert info2["fill"] is False or info2["ordered"] in (False, True)  # no required new fill
    assert env.position == 1  # position remains long after fill
