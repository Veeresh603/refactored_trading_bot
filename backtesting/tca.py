# backtesting/tca.py
"""
TCA helpers: fit simple power-law market impact model and estimate impact from params.

Model:
    impact = a * (size / adv) ** b

Functions:
- fit_powerlaw_impact(sizes, advs, impacts) -> {"a":..., "b":...}
- estimate_impact_from_params(size, adv, params) -> fractional impact (0..)
- calibrate_powerlaw(sizes, advs, impacts) -> returns (a,b) tuple (compat layer)
"""

from __future__ import annotations
from typing import Dict, Tuple
import math

import numpy as np

def _safe_log(x: np.ndarray) -> np.ndarray:
    """Numerically safe log (clip tiny positives)."""
    return np.log(np.clip(x, 1e-12, None))


def fit_powerlaw_impact(sizes: np.ndarray, advs: np.ndarray, impacts: np.ndarray) -> Dict[str, float]:
    """
    Fit impact = a * (size/adv)^b by linearizing log(impact) = log(a) + b * log(size/adv)
    Returns dict {"a": float, "b": float}.

    args are numpy arrays of matching length.
    """
    sizes = np.asarray(sizes, dtype=float)
    advs = np.asarray(advs, dtype=float)
    impacts = np.asarray(impacts, dtype=float)

    # sanity: require positive adv and sizes
    mask = (advs > 0) & (sizes > 0) & (impacts > 0)
    if mask.sum() < 2:
        # fallback default small impact
        return {"a": 0.0, "b": 0.0}

    x = sizes[mask] / advs[mask]
    y = impacts[mask]

    lx = _safe_log(x)
    ly = _safe_log(y)

    # linear least squares: ly = log(a) + b * lx
    A = np.vstack([np.ones_like(lx), lx]).T
    try:
        sol, *_ = np.linalg.lstsq(A, ly, rcond=None)
        loga, b = float(sol[0]), float(sol[1])
        a = float(math.exp(loga))
    except Exception:
        # fallback
        a, b = 0.0, 0.0

    # Ensure numeric scalars
    if not np.isfinite(a):
        a = 0.0
    if not np.isfinite(b):
        b = 0.0

    return {"a": float(a), "b": float(b)}


def estimate_impact_from_params(executed_units: float, bar_volume: float, params: Dict[str, float]) -> float:
    """
    Estimate fractional impact (e.g., 0.01 => 1% impact) using params {"a":..., "b":...}.
    If bar_volume==0 returns 0.0.
    """
    try:
        a = float(params.get("a", 0.0))
        b = float(params.get("b", 0.0))
    except Exception:
        return 0.0

    if bar_volume <= 0 or executed_units <= 0:
        return 0.0

    frac = (executed_units / float(bar_volume))
    # guard: negative/inf
    if not np.isfinite(frac) or frac <= 0.0:
        return 0.0

    try:
        return float(a * (frac ** b))
    except Exception:
        return 0.0


# convenience wrapper for older tests expecting tuple return
def calibrate_powerlaw(sizes: np.ndarray, advs: np.ndarray, impacts: np.ndarray) -> Tuple[float, float]:
    d = fit_powerlaw_impact(sizes, advs, impacts)
    return float(d.get("a", 0.0)), float(d.get("b", 0.0))
