"""
Strike Selector with Adaptive Logic
-----------------------------------
- Dynamically selects ATM/ITM/OTM strikes
- Uses volume + OI filters for liquidity
- Adapts based on implied volatility and market trend
- Requires broker.kite.instruments("NFO") loaded once
"""

import math
from datetime import date, timedelta


class StrikeSelector:
    def __init__(self, broker, lot_size=50, min_volume=1000, min_oi=5000):
        self.broker = broker
        self.lot_size = lot_size
        self.min_volume = min_volume
        self.min_oi = min_oi

    def _filter_liquid_strikes(self, option_chain, expiry):
        """Filter strikes by liquidity (volume & OI)."""
        return [
            opt for opt in option_chain
            if opt["expiry"] == expiry
            and opt["volume"] >= self.min_volume
            and opt["oi"] >= self.min_oi
        ]

    def _get_atm_strike(self, spot, step=50):
        """Round to nearest ATM strike."""
        return round(spot / step) * step

    def select_strikes(self, asset, spot, iv, expiry, step=50, trend="neutral"):
        """
        Adaptive strike selection.

        Args:
            asset (str): Asset name (e.g., NIFTY, BANKNIFTY)
            spot (float): Current spot price
            iv (float): Implied volatility (0.2 = 20%)
            expiry (date): Expiry date
            step (int): Strike step (default 50)
            trend (str): "bullish", "bearish", or "neutral"

        Returns:
            dict: Selected call/put strikes
        """
        atm = self._get_atm_strike(spot, step)

        # --- Adaptive logic ---
        if iv > 0.25:  # High volatility → safer OTM
            call_strike = atm + step
            put_strike = atm - step
        elif iv < 0.15:  # Very low IV → ATM
            call_strike, put_strike = atm, atm
        else:  # Moderate IV → slightly OTM
            call_strike, put_strike = atm + step, atm

        # Trend adjustment
        if trend == "bullish":
            call_strike += step  # more OTM calls
        elif trend == "bearish":
            put_strike -= step  # more OTM puts

        # --- Fetch option chain ---
        option_chain = self.broker.get_option_chain(asset)
        liquid_options = self._filter_liquid_strikes(option_chain, expiry)

        # Pick closest strikes available in chain
        selected_call = min(
            (opt for opt in liquid_options if opt["type"] == "CE"),
            key=lambda x: abs(x["strike"] - call_strike),
            default=None,
        )
        selected_put = min(
            (opt for opt in liquid_options if opt["type"] == "PE"),
            key=lambda x: abs(x["strike"] - put_strike),
            default=None,
        )

        return {
            "call": selected_call,
            "put": selected_put,
            "atm": atm,
            "iv": iv,
            "trend": trend,
        }
