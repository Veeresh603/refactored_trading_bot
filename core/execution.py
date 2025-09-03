"""
Execution Engine
----------------
- Places orders via broker
- Handles retries, slippage control, and iceberg orders
- Integrates with Risk Manager + Circuit Breaker
"""

import time
import logging
from core.risk_manager import AdvancedRiskManager
from core.circuit_breaker import CircuitBreaker


class ExecutionEngine:
    def __init__(self, broker, risk_manager=None, circuit_breaker=None,
                 max_retries=3, sleep_between_retries=2, slippage=0.001):
        """
        Args:
            broker: Broker object (core/broker.py)
            risk_manager: AdvancedRiskManager instance
            circuit_breaker: CircuitBreaker instance
            max_retries: Number of retries for failed orders
            sleep_between_retries: Delay between retries (sec)
            slippage: Slippage tolerance (fraction, e.g. 0.001 = 0.1%)
        """
        self.broker = broker
        self.risk_manager = risk_manager or AdvancedRiskManager()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.max_retries = max_retries
        self.sleep_between_retries = sleep_between_retries
        self.slippage = slippage
        self.last_order_time = None
        self.logger = logging.getLogger("ExecutionEngine")

    def _apply_slippage(self, price, action):
        """Adjust order price for slippage."""
        if action == "BUY":
            return price * (1 + self.slippage)
        elif action == "SELL":
            return price * (1 - self.slippage)
        return price

    def place_order(self, trade, account_equity=100000, position_size=0):
        """
        Place an order with retries and risk checks.

        Args:
            trade (dict): Trade order {symbol, qty, action, strike, is_call, expiry}
            account_equity (float): Current account equity
            position_size (int): Current open position size

        Returns:
            dict: Order response or failure reason
        """
        # Risk checks before placing
        pnl = trade.get("pnl", 0)
        safe, reason = self.risk_manager.check_risk(account_equity, pnl, position_size)
        if not safe:
            self.logger.error(f"❌ Risk Check Failed: {reason}")
            return {"status": "REJECTED", "reason": reason}

        cb_status = self.circuit_breaker.check(account_equity, pnl)
        if cb_status != "OK":
            self.logger.error(f"❌ Circuit Breaker Triggered: {cb_status}")
            return {"status": "REJECTED", "reason": cb_status}

        symbol = trade["symbol"]
        qty = trade["qty"]
        action = trade["action"]
        price = trade.get("price")  # optional
        if price:
            price = self._apply_slippage(price, action)

        order = None
        for attempt in range(1, self.max_retries + 1):
            try:
                order = self.broker.place_order(
                    symbol=symbol,
                    qty=qty,
                    side=action,
                    price=price,
                    product=trade.get("product", "MIS"),
                    order_type=trade.get("order_type", "MARKET"),
                )
                self.logger.info(f"✅ Order Placed: {action} {qty} {symbol} at {price or 'MKT'}")
                return {"status": "FILLED", "order": order}

            except Exception as e:
                self.logger.error(f"⚠️ Order attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.sleep_between_retries)
                else:
                    return {"status": "FAILED", "reason": str(e)}

        return {"status": "FAILED", "reason": "Unknown error"}

    def cancel_order(self, order_id):
        """Cancel an order safely."""
        try:
            result = self.broker.cancel_order(order_id)
            self.logger.info(f"🛑 Order cancelled: {order_id}")
            return result
        except Exception as e:
            self.logger.error(f"❌ Failed to cancel order {order_id}: {e}")
            return None
