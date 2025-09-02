"""
Advanced Risk Manager
---------------------
- Hedge Delta with futures
- Hedge Gamma & Vega with option spreads
"""

import math


class AdvancedRiskManager:
    def __init__(self, max_delta=100, max_gamma=0.002, max_vega=100, lot_size=50):
        self.max_delta = max_delta
        self.max_gamma = max_gamma
        self.max_vega = max_vega
        self.lot_size = lot_size

    # ---------------------------
    # Delta Hedging (Futures)
    # ---------------------------
    def hedge_with_futures(self, delta):
        lots = round(delta / self.lot_size)
        return lots  # +ve → SELL futures, -ve → BUY futures

    # ---------------------------
    # Gamma Hedging (Butterfly)
    # ---------------------------
    def hedge_gamma(self, asset, spot, expiry, step=50):
        """
        Hedge Gamma risk with a butterfly spread:
        - Buy 1 ITM option
        - Sell 2 ATM options
        - Buy 1 OTM option
        """
        atm = round(spot / step) * step
        strikes = [atm - step, atm, atm + step]
        return {
            "butterfly": [
                {"asset": asset, "expiry": expiry, "strike": strikes[0], "type": "CE", "qty": 1},   # Buy ITM
                {"asset": asset, "expiry": expiry, "strike": strikes[1], "type": "CE", "qty": -2},  # Sell ATM
                {"asset": asset, "expiry": expiry, "strike": strikes[2], "type": "CE", "qty": 1},   # Buy OTM
            ]
        }

    # ---------------------------
    # Vega Hedging (Straddle/Strangle)
    # ---------------------------
    def hedge_vega(self, asset, spot, expiry, step=50):
        """
        Hedge Vega with a long straddle:
        - Buy 1 ATM CE
        - Buy 1 ATM PE
        """
        atm = round(spot / step) * step
        return {
            "straddle": [
                {"asset": asset, "expiry": expiry, "strike": atm, "type": "CE", "qty": 1},  # Buy ATM CE
                {"asset": asset, "expiry": expiry, "strike": atm, "type": "PE", "qty": 1},  # Buy ATM PE
            ]
        }
