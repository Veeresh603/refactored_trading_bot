# backtesting/tca.py
"""
Basic TCA/impact parameter fitter.

We fit a simple power-law model:
    impact = a * (size / adv) ** b
Where `impact` is signed price movement observed (absolute adverse slippage),
`size` is execution size, and `adv` is average daily volume (or other normalizer).

Fitting is performed on logs:
    log(impact) = log(a) + b * log(size/adv)

This module provides a simple least-squares fit and a utility to estimate slippage
for a hypothetical execution share_of_liq.
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import numpy as np


def fit_powerlaw_impact(sizes: Iterable[float], advs: Iterable[float], impacts: Iterable[float]) -> Dict[str, float]:
    """
    Fit model: impact = a * (size / adv) ** b.
    sizes, advs, impacts must be same-length iterables, impacts > 0.
    Returns dict {a, b, r2}.
    """
    sizes = np.asarray(list(sizes), dtype=float)
    advs = np.asarray(list(advs), dtype=float)
    impacts = np.asarray(list(impacts), dtype=float)

    # filter valid positive
    mask = (sizes > 0) & (advs > 0) & (impacts > 0)
    if mask.sum() < 3:
        return {"a": 0.0, "b": 0.0, "r2": 0.0}
    x = np.log(sizes[mask] / advs[mask])
    y = np.log(impacts[mask])
    A = np.vstack([np.ones_like(x), x]).T
    coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    loga, b = coeffs[0], coeffs[1]
    a = float(np.exp(loga))
    # r2
    y_pred = loga + b * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"a": a, "b": float(b), "r2": r2}


def estimate_impact_from_params(size: float, adv: float, params: Dict[str, float]) -> float:
    """Return predicted impact (absolute price fraction) given params and normalization adv."""
    a = float(params.get("a", 0.0))
    b = float(params.get("b", 0.0))
    if size <= 0 or adv <= 0 or a == 0.0:
        return 0.0
    frac = (size / adv)
    return float(a * (frac ** b))
