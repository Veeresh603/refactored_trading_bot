import ctypes
import os
import pandas as pd
from datetime import datetime

# ----------------------------
# Load compiled C++ shared library
# ----------------------------
lib_path = os.path.join(os.path.dirname(__file__), "../cpp/execution_engine.so")
lib = ctypes.CDLL(lib_path)

# ----------------------------
# Global trade log
# ----------------------------
trade_log = []

# ----------------------------
# Order Placement (with logging)
# ----------------------------
lib.place_order.argtypes = [
    ctypes.c_char_p,  # symbol
    ctypes.c_int,     # qty
    ctypes.c_double,  # price
    ctypes.c_char_p,  # side
    ctypes.c_double,  # strike
    ctypes.c_double,  # sigma
    ctypes.c_bool,    # is_call
    ctypes.c_double   # expiry_days
]
lib.place_order.restype = ctypes.c_bool

def place_order(symbol, qty, price, side, strike=0, sigma=0.2, is_call=True, expiry_days=30):
    result = lib.place_order(symbol.encode(), qty, price, side.encode(),
                             strike, sigma, is_call, expiry_days)
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
lib.portfolio_greeks.argtypes = [
    ctypes.c_double,                     # spot price
    ctypes.POINTER(ctypes.c_double),     # delta
    ctypes.POINTER(ctypes.c_double),     # gamma
    ctypes.POINTER(ctypes.c_double),     # vega
    ctypes.POINTER(ctypes.c_double)      # theta
]

def portfolio_greeks(spot):
    d = ctypes.c_double()
    g = ctypes.c_double()
    v = ctypes.c_double()
    t = ctypes.c_double()
    lib.portfolio_greeks(spot, ctypes.byref(d), ctypes.byref(g), ctypes.byref(v), ctypes.byref(t))
    return d.value, g.value, v.value, t.value


# ----------------------------
# Hedging
# ----------------------------
lib.hedge_delta.argtypes = [ctypes.c_double, ctypes.c_int]
lib.hedge_delta.restype = ctypes.c_int

def hedge_delta(total_delta, lot_size=50):
    return lib.hedge_delta(total_delta, lot_size)

lib.hedge_gamma.argtypes = [ctypes.c_char_p, ctypes.c_double, ctypes.c_double]
lib.hedge_gamma.restype = None

def hedge_gamma(asset, atm, step=50):
    lib.hedge_gamma(asset.encode(), atm, step)

lib.hedge_vega.argtypes = [ctypes.c_char_p, ctypes.c_double]
lib.hedge_vega.restype = None

def hedge_vega(asset, atm):
    lib.hedge_vega(asset.encode(), atm)


# ----------------------------
# PnL Tracking
# ----------------------------
lib.unrealized_pnl.argtypes = [ctypes.c_double]
lib.unrealized_pnl.restype = ctypes.c_double

def unrealized_pnl(spot):
    return lib.unrealized_pnl(spot)

lib.total_pnl.argtypes = [ctypes.c_double]
lib.total_pnl.restype = ctypes.c_double

def total_pnl(spot):
    return lib.total_pnl(spot)


# ----------------------------
# Account Status
# ----------------------------
lib.account_status.argtypes = [
    ctypes.c_double,                     # spot
    ctypes.POINTER(ctypes.c_double),     # balance
    ctypes.POINTER(ctypes.c_double),     # margin_used
    ctypes.POINTER(ctypes.c_double),     # realized
    ctypes.POINTER(ctypes.c_double),     # unrealized
    ctypes.POINTER(ctypes.c_double)      # total
]
lib.account_status.restype = None

def account_status(spot):
    balance = ctypes.c_double()
    used_margin = ctypes.c_double()
    realized = ctypes.c_double()
    unrealized = ctypes.c_double()
    total = ctypes.c_double()

    lib.account_status(spot,
                       ctypes.byref(balance),
                       ctypes.byref(used_margin),
                       ctypes.byref(realized),
                       ctypes.byref(unrealized),
                       ctypes.byref(total))

    return {
        "balance": balance.value,
        "margin_used": used_margin.value,
        "realized": realized.value,
        "unrealized": unrealized.value,
        "total": total.value
    }


# ----------------------------
# Trade History
# ----------------------------
def get_trade_history(limit=20):
    """Return last N trades as DataFrame"""
    return pd.DataFrame(trade_log).tail(limit) if trade_log else pd.DataFrame(columns=["Time", "Symbol", "Side", "Qty", "Price"])
