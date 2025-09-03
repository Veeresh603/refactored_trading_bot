"""
Paper Trading Engine
--------------------
- Simulates live trading environment
- Tracks balance, margin, positions
- Calls execution_cpp for Greeks
- Logs trades for debugging
"""

import logging
from datetime import datetime
from core import execution_cpp

logger = logging.getLogger("PaperEngine")


class PaperTradingEngine:
    def __init__(self, initial_balance=100000, margin_perc=0.1):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.margin_perc = margin_perc
        self.trades = []
        self.reset()

        logger.info(f"📝 PaperTradingEngine initialized with balance={initial_balance}")

    def reset(self):
        execution_cpp.reset_engine(self.initial_balance)
        self.trades = []
        self.balance = self.initial_balance
        logger.info("🔄 Paper engine reset")

    def place_order(self, symbol, qty, price, side="BUY", strike=None, sigma=0.2, is_call=True, expiry_days=30):
        """
        Place a simulated order.

        Args:
            symbol (str): instrument symbol
            qty (int): quantity
            price (float): trade price
            side (str): BUY or SELL
            strike (float): option strike
            sigma (float): volatility
            is_call (bool): option type
            expiry_days (int): expiry horizon
        """
        try:
            if side == "BUY":
                execution_cpp.place_order(symbol, qty, price, strike or price, sigma, is_call, expiry_days)
                logger.info(f"📝 PAPER BUY {symbol} x{qty} @{price}")
            else:
                # For now, SELL = just placing another order (extend with netting logic if needed)
                execution_cpp.place_order(symbol, -qty, price, strike or price, sigma, is_call, expiry_days)
                logger.info(f"📝 PAPER SELL {symbol} x{qty} @{price}")

            trade = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "qty": qty,
                "price": price,
                "side": side,
                "strike": strike,
                "sigma": sigma,
                "is_call": is_call,
                "expiry_days": expiry_days,
            }
            self.trades.append(trade)

        except Exception as e:
            logger.error(f"❌ Paper order failed: {e}")

    def account_status(self, spot_price):
        """
        Wrapper for execution_cpp.account_status
        """
        return execution_cpp.account_status(spot_price)

    def portfolio_greeks(self, spot_price):
        """
        Wrapper for execution_cpp.portfolio_greeks
        """
        return execution_cpp.portfolio_greeks(spot_price)

    def get_trade_log(self):
        return self.trades
