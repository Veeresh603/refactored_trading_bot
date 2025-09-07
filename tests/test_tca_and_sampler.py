import numpy as np
import pytest

from backtesting.tca import estimate_impact_from_params, calibrate_powerlaw
from backtesting.orderbook_sampler import OrderBookSampler


def test_tca_estimation_and_calibration():
    # synthetic data
    executed = np.array([10, 20, 50, 100])
    volume = np.array([100, 100, 100, 100])
    true_a, true_b = 0.05, 0.6
    impact = true_a * (executed / volume) ** true_b

    a, b = calibrate_powerlaw(executed, volume, impact)
    assert a > 0 and b > 0
    est = estimate_impact_from_params(20, 100, {"a": a, "b": b})
    assert est > 0


def test_sampler_with_levels_and_df():
    # levels mode
    levels = [{"bids": [(99.0, 5.0)], "asks": [(101.0, 10.0)], "volume": 20.0}]
    s = OrderBookSampler(levels=levels, seed=42)
    res = s.execute(0, side=1, requested_units=2.0)
    assert res["executed"] == pytest.approx(2.0)
    assert res["vwap"] == pytest.approx(101.0)

    # df-like mode (list of dicts)
    df = [{"close": 100.0, "volume": 50.0}, {"close": 101.0, "volume": 50.0}]
    s2 = OrderBookSampler(df=df, liquidity_fraction=0.5, seed=123)
    res2 = s2.execute(1, side=-1, requested_units=20.0)
    assert res2["executed"] <= 25.0  # half liquidity
    assert res2["vwap"] > 0
    assert "latency_ms" in res2
