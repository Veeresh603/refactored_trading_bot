# smoke_check_env.py (use after you paste the new SimpleSyntheticEnv)
import math
from ai.train_rl import SimpleSyntheticEnv

def run_smoke():
    seed = 1234
    env = SimpleSyntheticEnv(window=3, episode_length=10,
                             price_drift=0.0, seed=seed,
                             fill_delay_steps=1, liquidity_fraction=1.0,
                             slippage_pct=0.0, commission=0.0)
    obs0 = env.reset()
    print("t0 idx:", env.idx, "last_price:", float(env.prices[env.idx]))
    # Enqueue buy (1) at t0
    obs1, reward1, done1, info1 = env.step(1)   # enqueue at idx=0, advance to idx=1 and process fill at 1
    print("After step(1): idx:", env.idx)
    print("info1:", info1)
    assert info1["ordered"] is True, "order should be enqueued"
    assert info1["fill"] is True, f"expected fill at idx {env.idx} but got info1"
    assert info1["fill_price"] is not None
    t0_price = float(env.prices[0])
    fill_price = info1["fill_price"]
    print("t0_price:", t0_price, "fill_price:", fill_price)
    assert not math.isclose(fill_price, t0_price), "Fill price must not equal t0 price (lookahead)"
    # Next step without new order: position should remain as filled
    obs2, reward2, done2, info2 = env.step(0)
    print("After step(0): idx:", env.idx, "info2:", info2)
    print("final position:", env.position, "final cash:", env.cash)
    assert env.position != 0.0, "Position should reflect filled order"
    print("SMOKE CHECK OK")

if __name__ == "__main__":
    run_smoke()
