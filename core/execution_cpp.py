import os
import sys
import pandas as pd
from datetime import datetime

# ----------------------------
# Ensure cpp/ folder is in Python path
# ----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
cpp_dir = os.path.join(base_dir, "../cpp")
sys.path.append(os.path.abspath(cpp_dir))

# ----------------------------
# Import compiled pybind11 module
# ----------------------------
try:
    import execution_engine  # built by setup.py
    print(f"✅ Loaded execution_engine from {cpp_dir}")
except ImportError as e:
    raise ImportError(
        "❌ Could not import execution_engine. "
        "Make sure you ran `python setup.py build_ext --inplace` inside cpp/"
    ) from e

# ----------------------------
# Global trade log
# ----------------------------
trade_log = []


# ----------------------------
# Order Placement
# ----------------------------
def place_order(symbol, qty, price, side, strike=0, sigma=0.2, is_call=True, expiry_days=30):
    result = execution_engine.place_order(symbol, qty, price, side, strike, sigma, is_call, expiry_days)
    if result:
        trade_log.append({
            "Time": datetime.now(),
            "Symbol": symbol,
            "Side": side,
            "Qty": qty,
            "Price": price
        })
    return result


# ----------------------------
# Portfolio Greeks
# ----------------------------
def portfolio_greeks(spot):
    return execution_engine.portfolio_greeks(spot)


# ----------------------------
# Hedging
# ----------------------------
def hedge_delta(total_delta, lot_size=50):
    return execution_engine.hedge_delta(total_delta, lot_size)


def hedge_gamma(asset, atm, step=50):
    return execution_engine.hedge_gamma(asset, atm, step)


def hedge_vega(asset, atm):
    return execution_engine.hedge_vega(asset, atm)


# ----------------------------
# PnL Tracking
# ----------------------------
def unrealized_pnl(spot):
    return execution_engine.unrealized_pnl(spot)


def total_pnl(spot):
    return execution_engine.total_pnl(spot)


# ----------------------------
# Account Status
# ----------------------------
def account_status(spot):
    return execution_engine.account_status(spot)


# ----------------------------
# Trade History
# ----------------------------
def get_trade_history(limit=20):
    """Return last N trades as DataFrame"""
    return pd.DataFrame(trade_log).tail(limit) if trade_log else pd.DataFrame(
        columns=["Time", "Symbol", "Side", "Qty", "Price"]
    )
