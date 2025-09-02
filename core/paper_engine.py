import logging
from core import execution_cpp

logger = logging.getLogger("PaperTrading")

class PaperTradingEngine:
    def __init__(self, initial_balance=2_000_000):
        self.initial_balance = initial_balance
        logger.info(f"📝 Paper Trading Engine initialized with balance={initial_balance}")

    def place_order(self, symbol, qty, price, side, strike=0, sigma=0.2, is_call=True, expiry_days=30):
        """Simulated order execution"""
        result = execution_cpp.place_order(symbol, qty, price, side, strike, sigma, is_call, expiry_days)
        if result:
            logger.info(f"📝 Paper order {side} {qty} {symbol} @ {price}")
        else:
            logger.warning(f"⚠️ Paper order rejected for {symbol}")
        return result

    def get_greeks(self, spot):
        return execution_cpp.portfolio_greeks(spot)

    def get_account_status(self, spot):
        return execution_cpp.account_status(spot)

    def get_pnl(self, spot):
        return execution_cpp.total_pnl(spot)

    def get_trade_history(self, limit=20):
        return execution_cpp.get_trade_history(limit)
