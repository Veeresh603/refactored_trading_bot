# core/execution_engine.py
import logging
logger = logging.getLogger("TradingBot.execution_engine")

try:
    import execution_engine_cpp as _cpp  # compiled module from pybind11
    logger.info("✅ Loaded execution_engine_cpp extension")
except Exception as e:
    _cpp = None
    logger.warning(f"⚠️ Could not load execution_engine_cpp: {e}")

# wrapper API that main.py expects
def reset_engine(init_balance: float):
    if _cpp and hasattr(_cpp, "reset_engine"):
        return _cpp.reset_engine(init_balance)
    logger.warning("⚠️ reset_engine not available; using python fallback")
    # fallback: create a simple python engine state in this module
    global _py_state
    _py_state = {"balance": init_balance, "positions": []}
    return True

def place_order(symbol, qty, price, strike, sigma, is_call, expiry_days):
    if _cpp and hasattr(_cpp, "place_order"):
        return _cpp.place_order(symbol, qty, price, strike, sigma, is_call, expiry_days)
    logger.warning("⚠️ place_order not available; recording in python fallback")
    pos = {
        "symbol": symbol,
        "qty": qty,
        "price": price,
        "strike": strike,
        "sigma": sigma,
        "is_call": is_call,
        "expiry_days": expiry_days
    }
    _py_state.setdefault("positions", []).append(pos)
    logger.info(f"[FALLBACK] Placed order: {symbol} qty={qty} @ {price} expiry_days={expiry_days} sigma={sigma}")

def account_status(spot):
    if _cpp and hasattr(_cpp, "account_status"):
        return _cpp.account_status(spot)
    logger.warning("⚠️ account_status not available; returning fallback zeros")
    return {"balance": _py_state.get("balance", 0.0),
            "total": _py_state.get("balance", 0.0),
            "realized": 0.0,
            "unrealized": 0.0}

def portfolio_greeks(spot):
    if _cpp and hasattr(_cpp, "portfolio_greeks"):
        try:
            return _cpp.portfolio_greeks(spot)
        except Exception as e:
            logger.warning(f"⚠️ error calling portfolio_greeks in C++ module: {e}")
    # fallback: zeros (safe)
    logger.warning("⚠️ portfolio_greeks missing; returning zeros")
    return (0.0, 0.0, 0.0, 0.0)

def get_positions():
    if _cpp and hasattr(_cpp, "get_positions"):
        return _cpp.get_positions()
    return _py_state.get("positions", [])

def get_trade_log():
    if _cpp and hasattr(_cpp, "get_trade_log"):
        return _cpp.get_trade_log()
    return []
