"""
Python wrapper around the compiled C++ execution_engine_cpp module
"""

from execution_engine_cpp import ExecutionEngine  # ✅ renamed module

# Global engine instance
_engine = ExecutionEngine()


def reset_engine(initial_balance: float):
    """Reset global execution engine."""
    return _engine.reset(initial_balance)


def place_order(symbol, qty, price, strike, sigma, is_call, expiry_days):
    """Place an order via the global engine."""
    return _engine.place_order(symbol, qty, price, strike, sigma, is_call, expiry_days)


def account_status(spot_price):
    """Get account status from global engine."""
    return _engine.account_status(spot_price)


def portfolio_greeks(spot_price):
    """Get portfolio Greeks from global engine."""
    return _engine.portfolio_greeks(spot_price)


def get_trade_log():
    """Return trade log from global engine."""
    return _engine.get_trade_log()
