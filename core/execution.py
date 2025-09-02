"""
Execution Engine
----------------
- Places orders via broker
- Handles retries, slippage control, and iceberg orders
- Central point for trade execution
"""

import time


class ExecutionEngine:
    def __init__(self, broker, max_retries=3, sleep_between_retries=2):
        """
        broker: Broker object (core/broker.py)
        max_retries: number of times to retry failed order
        """
        self.broker = broker
        self.max_retries = max_retries
        self.sleep_between_retries = sleep_between_retries

    # -------------------------------
    # Core Execution
    # -------------------------------
    def execute_trade(self, symbol, qty, direction="BUY", price=None, iceberg_size=None):
        """
        Execute a trade with retry and optional iceberg splitting
        """
        if iceberg_size and qty > iceberg_size:
            return self._execute_iceberg(symbol, qty, direction, price, iceberg_size)

        return self._place_with_retry(symbol, qty, direction, price)

    def _place_with_retry(self, symbol, qty, direction, price):
        """
        Place order with retries
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.broker.place_order(symbol, qty, direction, price)
                if response.get("status") == "success":
                    print(f"✅ Order successful: {response}")
                    return response
            except Exception as e:
                print(f"❌ Order attempt {attempt} failed: {e}")
                time.sleep(self.sleep_between_retries)

        return {"status": "error", "message": f"Failed after {self.max_retries} attempts"}

    # -------------------------------
    # Iceberg Orders
    # -------------------------------
    def _execute_iceberg(self, symbol, qty, direction, price, iceberg_size):
        """
        Split a large order into smaller chunks
        """
        print(f"🧊 Executing iceberg order: {qty} in chunks of {iceberg_size}")

        responses = []
        chunks = qty // iceberg_size
        remainder = qty % iceberg_size

        for i in range(chunks):
            resp = self._place_with_retry(symbol, iceberg_size, direction, price)
            responses.append(resp)

        if remainder > 0:
            resp = self._place_with_retry(symbol, remainder, direction, price)
            responses.append(resp)

        return responses
