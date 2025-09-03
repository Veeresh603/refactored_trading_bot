"""
Options Optimizer
-----------------
- Converts strategy decision into specific option contract
- Handles strike selection, expiry choice, lot sizing
"""

import logging
from datetime import datetime, timedelta

logger = logging.getLogger("OptionsOptimizer")


class OptionsOptimizer:
    def __init__(self, config, lot_size=50):
        """
        Args:
            config (dict): from config.yaml
            lot_size (int): default lot size per trade
        """
        self.config = config
        self.lot_size = lot_size

    def evaluate(self, decision):
        """
        Build option contract from strategy decision.

        Args:
            decision (dict): must contain
                - asset (str)
                - spot (float)
                - signal (1=Buy Call, -1=Buy Put, 0=Hold)
                - strike_offset (int)
                - expiry (str: weekly | monthly)
                - greeks (dict)
                - iv (float)

        Returns:
            dict with contract details
        """
        asset = decision["asset"]
        spot = decision["spot"]
        signal = decision.get("signal", 0)
        strike_offset = decision.get("strike_offset", 0)
        expiry_type = decision.get("expiry", "weekly")
        iv = decision.get("iv", 0.2)

        if signal == 0:
            logger.info("⚖️ No trade signal, staying flat")
            return None

        # --- Strike Selection ---
        base_strike = round(spot / 50) * 50  # ATM rounded to nearest 50
        strike = base_strike + strike_offset

        # --- Expiry Selection ---
        today = datetime.now().date()
        if expiry_type == "weekly":
            expiry = today + timedelta(days=7 - today.weekday())  # next Thursday
        else:
            next_month = (today.month % 12) + 1
            expiry = datetime(today.year if next_month != 1 else today.year + 1, next_month, 28).date()

        # --- Build Symbol ---
        option_type = "CE" if signal == 1 else "PE"
        symbol = f"{asset}{strike}{option_type}"

        contract = {
            "symbol": symbol,
            "asset": asset,
            "strike": strike,
            "expiry": expiry,
            "is_call": (signal == 1),
            "qty": self.lot_size,
            "iv": iv,
        }

        logger.info(f"📦 Built contract: {contract}")
        return contract
