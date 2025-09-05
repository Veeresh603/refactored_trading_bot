# tests/test_tca_fit.py
from backtesting.tca import fit_powerlaw_impact, estimate_impact_from_params
import numpy as np

def test_fit_powerlaw_and_estimate():
    # synthetic sizes/advs and impacts generated from a known a,b
    a_true = 0.02
    b_true = 0.6
    adv = np.array([1000, 2000, 5000, 10000], dtype=float)
    sizes = np.array([10, 50, 100, 200], dtype=float)
    impacts = a_true * (sizes / adv) ** b_true
    params = fit_powerlaw_impact(sizes, adv, impacts)
    assert params["a"] > 0
    assert 0.0 < params["b"] < 2.0
    est = estimate_impact_from_params(50, 2000, params)
    assert est >= 0.0
