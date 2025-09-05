# core/execution_engine.py
"""
Execution engine (Python fallback) with configurable market impact & slicing.

Features:
- MARKET and LIMIT handling (immediate-fill conservative model)
- Optional slicing (TWAP-like) for large orders
- Temporary & permanent impact modeling (configurable coefficients)
- Per-symbol ADV estimate to normalize impact (set via state or during runtime)
- Thread-safe and logging friendly
- Compatible with the previous lightweight API (reset_engine, place_order, cancel_order, process_market_tick, get_positions, get_trade_log)
"""

from __future__ import annotations

import threading
import time
import logging
import itertools
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("TradingBot.execution_engine")
logger.setLevel(logging.INFO)

try:
    import execution_engine_cpp as _cpp
    logger.info("Loaded execution_engine_cpp extension")
except Exception as e:
    _cpp = None
    logger.info(f"execution_engine_cpp not available, using Python fallback: {e}")

_py_lock = threading.RLock()
_py_state: Dict[str, Any] = {
    "balance": 100000.0,
    "cash": 100000.0,
    "positions": {},  # symbol -> {qty, avg_price}
    "trade_log": [],
    "order_id_seq": itertools.count(1),
    "fees_per_trade": 1.0,
    "default_slippage": 0.0005,  # fraction
    # market impact settings
    "temp_impact_coeff": 0.1,  # temporary impact base coefficient (configurable)
    "perm_impact_coeff": 0.02,  # permanent impact base coefficient (configurable)
    "adv_estimates": {},  # symbol -> adv (shares) estimate for normalizing impact
    "slice_threshold": 1000,  # if qty > threshold, automatically slice
    "default_slice_count": 5,
}


@dataclass
class Fill:
    order_id: int
    symbol: str
    qty: int
    side: str
    price: float
    timestamp: float
    fees: float
    slippage: float
    status: str = "FILLED"
    slice_index: Optional[int] = None
    slice_count: Optional[int] = None


# -------------------------
# Helpers: market impact model
# -------------------------
def _get_adv(symbol: str) -> float:
    with _py_lock:
        adv = _py_state["adv_estimates"].get(symbol, None)
    if adv is None or adv <= 0:
        # fallback to a large number so impact is moderate for small synthetic runs
        return 1_000_000.0
    return float(adv)


def _temporary_impact(slice_qty: int, adv: float) -> float:
    """
    Simple temporary impact model:
      temp = temp_coeff * (slice_qty / adv)^0.5
    This is a flexible empirical model (user can tune coefficients).
    """
    coeff = float(_py_state.get("temp_impact_coeff", 0.1))
    ratio = max(0.0, float(slice_qty) / max(1.0, adv))
    return coeff * (ratio ** 0.5)


def _permanent_impact(slice_qty: int, adv: float) -> float:
    """
    Simple permanent impact model:
      perm = perm_coeff * (slice_qty / adv)
    """
    coeff = float(_py_state.get("perm_impact_coeff", 0.02))
    ratio = max(0.0, float(slice_qty) / max(1.0, adv))
    return coeff * ratio


# -------------------------
# Public API
# -------------------------
def reset_engine(init_balance: float = 100000.0):
    if _cpp and hasattr(_cpp, "reset_engine"):
        return _cpp.reset_engine(init_balance)
    with _py_lock:
        _py_state["balance"] = float(init_balance)
        _py_state["cash"] = float(init_balance)
        _py_state["positions"] = {}
        _py_state["trade_log"] = []
        _py_state["last_market"] = {}
        _py_state["order_id_seq"] = itertools.count(1)
    logger.info(f"Execution engine reset: balance={init_balance}")
    return True


def process_market_tick(symbol: str, price: float, timestamp: Optional[float] = None):
    if _cpp and hasattr(_cpp, "process_market_tick"):
        return _cpp.process_market_tick(symbol, price, timestamp)
    with _py_lock:
        _py_state["last_market"][symbol] = {"price": float(price), "timestamp": timestamp or time.time()}
    return True


def _apply_fill(symbol: str, qty: int, side: str, exec_price: float, fees: float, slippage_frac: float,
                order_id: int, slice_index: Optional[int] = None, slice_count: Optional[int] = None):
    """Update positions and cash for a single fill."""
    with _py_lock:
        notional = exec_price * qty
        slippage_cost = abs(notional) * slippage_frac
        if side.upper() == "BUY":
            if _py_state["cash"] < (notional + fees + slippage_cost):
                raise RuntimeError("INSUFFICIENT_CASH")
            _py_state["cash"] -= (notional + fees + slippage_cost)
            pos = _py_state["positions"].get(symbol, {"qty": 0, "avg_price": 0.0})
            prev_qty = pos["qty"]
            prev_avg = pos["avg_price"]
            new_qty = prev_qty + qty
            new_avg = ((prev_avg * prev_qty) + (exec_price * qty)) / new_qty if new_qty != 0 else 0.0
            _py_state["positions"][symbol] = {"qty": new_qty, "avg_price": new_avg}
        else:
            pos = _py_state["positions"].get(symbol, {"qty": 0, "avg_price": 0.0})
            prev_qty = pos["qty"]
            if qty > prev_qty:
                # conservative: do not allow naked shorts in fallback
                raise RuntimeError("INSUFFICIENT_POSITION")
            proceeds = exec_price * qty
            _py_state["cash"] += (proceeds - fees - abs(proceeds) * slippage_frac)
            new_qty = prev_qty - qty
            if new_qty == 0:
                _py_state["positions"].pop(symbol, None)
            else:
                _py_state["positions"][symbol] = {"qty": new_qty, "avg_price": pos.get("avg_price", 0.0)}
        # append trade log
        fill = Fill(order_id=order_id, symbol=symbol, qty=qty, side=side.upper(),
                    price=exec_price, timestamp=time.time(), fees=fees, slippage=slippage_frac,
                    slice_index=slice_index, slice_count=slice_count)
        _py_state["trade_log"].append(asdict(fill))
        # update balance
        mv = 0.0
        for s_sym, s_pos in _py_state["positions"].items():
            last_price = _py_state["last_market"].get(s_sym, {}).get("price", 0.0)
            mv += s_pos.get("qty", 0) * last_price
        _py_state["balance"] = _py_state["cash"] + mv
    return asdict(fill)


def place_order(symbol: str, qty: int, side: str = "BUY", order_type: str = "MARKET",
                price: Optional[float] = None, slippage: Optional[float] = None,
                client_order_id: Optional[str] = None,
                slice_count: Optional[int] = None, slice_threshold: Optional[int] = None) -> Dict[str, Any]:
    """
    Place an order with optional slicing and impact modelling.

    - If qty exceeds slice_threshold (or global threshold), the order will be split into slice_count pieces.
    - Temporary & permanent impact are applied per slice, with ADV normalization.
    """
    if _cpp and hasattr(_cpp, "place_order"):
        return _cpp.place_order(symbol, qty, side, order_type, price, slippage, client_order_id)

    with _py_lock:
        order_id = next(_py_state["order_id_seq"])
        last = _py_state["last_market"].get(symbol)
        if last is None:
            reason = "NO_MARKET_PRICE"
            logger.warning(f"Order {order_id} for {symbol} rejected: no market price available")
            return {"status": "REJECTED", "reason": reason, "order_id": order_id}

        market_price = float(last["price"])
        adv = _get_adv(symbol)

        # Determine slicing
        global_threshold = int(_py_state.get("slice_threshold", 1000))
        if slice_threshold is None:
            slice_threshold = global_threshold
        if slice_count is None:
            slice_count = int(_py_state.get("default_slice_count", 5))

        abs_qty = abs(int(qty))
        do_slice = abs_qty > slice_threshold and slice_count > 1

        # simple limit logic: only fill if price crosses for LIMIT
        if order_type.upper() == "LIMIT":
            if (side.upper() == "BUY" and price >= market_price) or (side.upper() == "SELL" and price <= market_price):
                # treat as filled at the limit price (apply small slippage)
                exec_price = float(price)
                slippage_frac = float(slippage if slippage is not None else _py_state["default_slippage"])
                fees = float(_py_state.get("fees_per_trade", 1.0))
                try:
                    fill = _apply_fill(symbol, abs_qty, side, exec_price, fees, slippage_frac, order_id)
                    logger.info(f"Limit order filled id={order_id} {side} {qty}@{exec_price} {symbol}")
                    return {"status": "FILLED", "order_id": order_id, "fill": fill, "balance": _py_state["balance"]}
                except RuntimeError as e:
                    return {"status": "REJECTED", "reason": str(e), "order_id": order_id}
            else:
                return {"status": "NOT_FILLED", "reason": "LIMIT_NOT_HIT", "order_id": order_id}

        # MARKET or immediate-fill with slicing & impact
        if order_type.upper() == "MARKET":
            fees = float(_py_state.get("fees_per_trade", 1.0))
            slippage_frac_base = float(slippage if slippage is not None else _py_state["default_slippage"])
            # If slicing, iterate slices
            if do_slice:
                slice_qty = abs_qty // slice_count
                leftover = abs_qty - slice_qty * slice_count
                fills = []
                cumulative_price = market_price
                for i in range(slice_count):
                    this_qty = slice_qty + (1 if i < leftover else 0)
                    # compute impact
                    temp = _temporary_impact(this_qty, adv)
                    perm = _permanent_impact(this_qty, adv)
                    # temporary slippage adds to execution price for BUY, subtracts for SELL
                    exec_price = cumulative_price * (1 + (temp + slippage_frac_base) if side.upper() == "BUY" else 1 - (temp + slippage_frac_base))
                    # apply fill
                    try:
                        fill = _apply_fill(symbol, this_qty, side, exec_price, fees, temp + slippage_frac_base, order_id, slice_index=i+1, slice_count=slice_count)
                    except RuntimeError as e:
                        logger.warning(f"Slice {i+1} rejected: {e}")
                        return {"status": "REJECTED", "reason": str(e), "order_id": order_id}
                    fills.append(fill)
                    # permanent impact shifts the mid-price for subsequent slices
                    cumulative_price = cumulative_price * (1 + perm if side.upper() == "BUY" else 1 - perm)
                logger.info(f"Order FILLED (sliced): id={order_id} {side} {qty}@~{market_price} symbol={symbol} slices={slice_count}")
                return {"status": "FILLED", "order_id": order_id, "fills": fills, "balance": _py_state["balance"]}
            else:
                # single-fill market order with temporary slippage
                temp = _temporary_impact(abs_qty, adv)
                slippage_frac = temp + slippage_frac_base
                exec_price = market_price * (1 + slippage_frac if side.upper() == "BUY" else 1 - slippage_frac)
                try:
                    fill = _apply_fill(symbol, abs_qty, side, exec_price, fees, slippage_frac, order_id, slice_index=1, slice_count=1)
                except RuntimeError as e:
                    return {"status": "REJECTED", "reason": str(e), "order_id": order_id}
                logger.info(f"Order FILLED: id={order_id} {side} {qty}@{exec_price} {symbol} (temp_imp={temp:.6f})")
                return {"status": "FILLED", "order_id": order_id, "fill": fill, "balance": _py_state["balance"]}
        else:
            return {"status": "REJECTED", "reason": "UNKNOWN_ORDER_TYPE", "order_id": order_id}


def cancel_order(order_id: int) -> Dict[str, Any]:
    if _cpp and hasattr(_cpp, "cancel_order"):
        return _cpp.cancel_order(order_id)
    logger.info(f"cancel_order called for {order_id} — no-op in immediate-fill fallback")
    return {"status": "NOT_FOUND", "order_id": order_id}


def portfolio_greeks() -> Tuple[float, float, float, float]:
    if _cpp and hasattr(_cpp, "portfolio_greeks"):
        try:
            return _cpp.portfolio_greeks()
        except Exception as e:
            logger.warning(f"error calling portfolio_greeks in C++ module: {e}")
    return (0.0, 0.0, 0.0, 0.0)


def get_positions() -> List[Dict[str, Any]]:
    if _cpp and hasattr(_cpp, "get_positions"):
        return _cpp.get_positions()
    with _py_lock:
        return [{**{"symbol": k}, **v} for k, v in _py_state.get("positions", {}).items()]


def get_trade_log() -> List[Dict[str, Any]]:
    if _cpp and hasattr(_cpp, "get_trade_log"):
        return _cpp.get_trade_log()
    with _py_lock:
        return list(_py_state.get("trade_log", []))


# -------------------------
# Management APIs (runtime tuning)
# -------------------------
def set_adv_estimate(symbol: str, adv: float):
    with _py_lock:
        _py_state["adv_estimates"][symbol] = float(adv)


def set_impact_coeffs(temp_coeff: float = None, perm_coeff: float = None):
    with _py_lock:
        if temp_coeff is not None:
            _py_state["temp_impact_coeff"] = float(temp_coeff)
        if perm_coeff is not None:
            _py_state["perm_impact_coeff"] = float(perm_coeff)
