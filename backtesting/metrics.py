# backtesting/metrics.py
"""
Backtesting metrics helpers.

Provides:
- _calc_sharpe(returns, period=252)  # legacy/private name expected by some modules
- _calc_max_drawdown(equity_series)  # legacy/private name expected by some modules
- compute_metrics(...)  # compatibility wrapper expected by older code

Also exposes public wrappers:
- calc_sharpe(returns, period=252)
- calc_max_drawdown(equity_series)

Notes:
- `returns` may be a numpy array or pandas Series of periodic returns (not cumulative).
- `equity_series` may be a pandas Series or numpy array of equity values over time.
- Sharpe returned is annualized (assuming `period` observations per year).
- Max drawdown returned as fraction (e.g., 0.2 => 20% drawdown).
"""

from __future__ import annotations
from typing import Union, Sequence, Tuple, Optional, Any, Dict
import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, pd.Series, Sequence[float]]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float)
    return np.asarray(x, dtype=float)


# -------------------------
# Max drawdown
# -------------------------
def _calc_max_drawdown(equity_series: ArrayLike) -> float:
    """
    Compute maximum drawdown fraction from an equity curve.

    Args:
        equity_series: array-like of equity values (time-ordered)

    Returns:
        max_drawdown (float) in [0, 1), where 0 means no drawdown.
    """
    eq = _to_numpy(equity_series).astype(float)
    if eq.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(eq)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(running_max > 0.0, (running_max - eq) / running_max, 0.0)
    max_dd = float(np.nanmax(dd)) if dd.size > 0 else 0.0
    if not np.isfinite(max_dd):
        return 0.0
    return max_dd


# -------------------------
# Sharpe ratio (simple)
# -------------------------
def _calc_sharpe(returns: ArrayLike, period: int = 252) -> float:
    """
    Compute (annualized) Sharpe ratio from periodic returns.

    Args:
        returns: array-like of periodic returns (e.g., daily returns)
        period: number of periods per year for annualization (default 252 for daily)

    Returns:
        Annualized Sharpe ratio (float). If returns are constant or zero variance,
        returns 0.0 to indicate no signal.
    """
    r = _to_numpy(returns).astype(float)
    if r.size == 0:
        return 0.0

    # Heuristic: if caller passed equity/prices instead of returns, convert
    # (detect by checking for many positive monotonic numbers with large range)
    if r.size > 1 and (np.nanmax(r) - np.nanmin(r) > 1.0) and np.all(np.isfinite(r)):
        # try converting to pct-change; use safe handling
        r_conv = np.diff(r) / (r[:-1] + 1e-12)
        # acceptable only if conversion yields non-NaN values
        if r_conv.size > 0 and np.isfinite(r_conv).all():
            r = r_conv

    mean = float(np.nanmean(r))
    std = float(np.nanstd(r, ddof=1)) if r.size > 1 else 0.0
    if std <= 0.0 or not np.isfinite(std):
        return 0.0
    sharpe = (mean / std) * np.sqrt(float(period))
    if not np.isfinite(sharpe):
        return 0.0
    return float(sharpe)


# -------------------------
# Public aliases (stable API)
# -------------------------
def calc_max_drawdown(equity_series: ArrayLike) -> float:
    return _calc_max_drawdown(equity_series)


def calc_sharpe(returns: ArrayLike, period: int = 252) -> float:
    return _calc_sharpe(returns, period=period)


# -------------------------
# Convenience: combined metrics for a backtest
# -------------------------
def summary_metrics(equity_series: ArrayLike, returns: Optional[ArrayLike] = None, period: int = 252) -> Dict[str, Any]:
    """
    Compute a small set of common backtest metrics:
      - final_equity
      - total_return
      - n_periods
      - n_trades (placeholder 0.0 if unavailable)
      - max_drawdown
      - sharpe

    returns: if None, compute returns from equity_series by pct change.
    """
    eq = _to_numpy(equity_series).astype(float)
    final_equity = float(eq[-1]) if eq.size > 0 else 0.0
    start_eq = float(eq[0]) if eq.size > 0 else 0.0
    total_return = float((final_equity - start_eq) / (start_eq + 1e-12)) if eq.size > 0 else 0.0
    n_periods = int(eq.size)
    if returns is None:
        if eq.size > 1:
            ret = np.diff(eq) / (eq[:-1] + 1e-12)
        else:
            ret = np.array([], dtype=float)
    else:
        ret = _to_numpy(returns).astype(float)

    max_dd = _calc_max_drawdown(eq)
    sharpe = _calc_sharpe(ret, period=period)

    return {
        "final_equity": final_equity,
        "total_return": total_return,
        "n_periods": n_periods,
        "n_trades": 0.0,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
    }


# -------------------------
# Backwards-compatible wrapper expected in some modules
# -------------------------
def compute_metrics(*args, **kwargs) -> Dict[str, Any]:
    """
    Backwards-compatible alias for older code that expects `compute_metrics`.
    Internally forwards to summary_metrics; accepts same arguments.
    """
    return summary_metrics(*args, **kwargs)


# Exported names
__all__ = [
    "calc_sharpe",
    "calc_max_drawdown",
    "_calc_sharpe",
    "_calc_max_drawdown",
    "summary_metrics",
    "compute_metrics",
]
