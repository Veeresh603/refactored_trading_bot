# backtesting/orderbook_adapter.py
"""
Production-grade OrderBook adapter:

- Defensively calls an orderbook sampler's execute(...) and converts/validates the response.
- Returns a consistent dict with keys:
    {"executed": float, "vwap": Optional[float], "liquidity_used": float, "latency_ms": float, "raw": original_res}
- Handles dict / object responses and malformed values (strings, nan, None).
- Logs warnings (no exceptions bubble to caller).
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import math
import logging

logger = logging.getLogger("TradingBot.OrderBookAdapter")


def _is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def _to_float_safe(x: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Convert to float safely. Recognize 'nan', 'inf', None and return default.
    If default is None, returns None on invalids.
    """
    if x is None:
        return default
    try:
        # strings like 'nan' -> float('nan') -> nan, we filter
        v = float(x)
    except Exception:
        # Try to handle string variants explicitly
        try:
            s = str(x).strip().lower()
            if s in ("nan", "na", "none", ""):
                return default
            if s in ("inf", "+inf", "-inf", "infinity"):
                return default
        except Exception:
            pass
        return default
    if not _is_finite(v):
        return default
    return v


def execute_safe(orderbook_sampler: Any, idx: int, side: int, requested_units: float) -> Dict[str, Any]:
    """
    Call sampler.execute and coerce results to a robust dictionary.
    Always returns a dict with numeric executed & liquidity_used (floats), vwap (float|None), latency_ms (float).
    The original raw response is included under 'raw' for debugging.
    """
    # default return
    safe_default = {"executed": 0.0, "vwap": None, "liquidity_used": 0.0, "latency_ms": 0.0, "raw": None}
    if orderbook_sampler is None:
        return safe_default

    try:
        res = orderbook_sampler.execute(int(idx), side=int(side), requested_units=float(requested_units))
    except Exception as exc:
        logger.exception("OrderBookSampler.execute raised at idx=%s: %s", idx, exc)
        return safe_default

    safe = {"raw": res, "latency_ms": 0.0}
    if not res:
        # None-like response
        safe.update({"executed": 0.0, "vwap": None, "liquidity_used": 0.0})
        return safe

    # unify dict-like or object-like response
    if isinstance(res, dict):
        executed_raw = res.get("executed", res.get("filled", 0.0))
        vwap_raw = res.get("vwap", res.get("price", None))
        liq_raw = res.get("liquidity_used", res.get("liquidity", res.get("filled", 0.0)))
        latency_raw = res.get("latency_ms", res.get("latency", 0.0))
    else:
        executed_raw = getattr(res, "executed", getattr(res, "filled", 0.0))
        vwap_raw = getattr(res, "vwap", getattr(res, "price", None))
        liq_raw = getattr(res, "liquidity_used", getattr(res, "liquidity", getattr(res, "filled", 0.0)))
        latency_raw = getattr(res, "latency_ms", getattr(res, "latency", 0.0))

    executed = _to_float_safe(executed_raw, default=0.0) or 0.0
    if executed < 0.0:
        logger.warning("OrderBookSampler returned negative executed=%s at idx=%s; clamping to 0", executed_raw, idx)
        executed = 0.0

    vwap_val = _to_float_safe(vwap_raw, default=None)
    if vwap_raw is not None and vwap_val is None:
        logger.warning("OrderBookSampler returned non-numeric vwap=%s at idx=%s; setting vwap=None", vwap_raw, idx)

    liquidity_used = _to_float_safe(liq_raw, default=0.0) or 0.0
    if liquidity_used < 0.0:
        logger.warning("OrderBookSampler returned negative liquidity_used=%s at idx=%s; clamping to 0", liq_raw, idx)
        liquidity_used = 0.0

    latency_ms = _to_float_safe(latency_raw, default=0.0) or 0.0
    if latency_ms < 0.0:
        latency_ms = 0.0

    safe.update({
        "executed": float(executed),
        "vwap": float(vwap_val) if vwap_val is not None else None,
        "liquidity_used": float(liquidity_used),
        "latency_ms": float(latency_ms),
    })
    return safe
