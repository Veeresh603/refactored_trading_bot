"""
Future Selector
---------------
- Selects futures contract for a given asset
- Adaptive expiry selection (near/far month based on volatility & liquidity)
- Auto-rollover near expiry
"""

from datetime import date, timedelta


class FutureSelector:
    def __init__(self, broker, min_volume=1000, min_oi=5000, rollover_days=2):
        self.broker = broker
        self.instruments = broker.kite.instruments("NFO")
        self.min_volume = min_volume
        self.min_oi = min_oi
        self.rollover_days = rollover_days

    def _filter_liquid_futures(self, futures):
        """Filter futures contracts by liquidity."""
        return [
            f for f in futures
            if f.get("volume", 0) >= self.min_volume and f.get("oi", 0) >= self.min_oi
        ]

    def select_future(self, asset, iv=0.2, prefer_far=False):
        """
        Select the best futures contract.

        Args:
            asset (str): Asset name (e.g., NIFTY, BANKNIFTY)
            iv (float): Implied volatility (default 0.2)
            prefer_far (bool): Force far-month selection

        Returns:
            dict: Selected future instrument
        """
        futures = [
            inst for inst in self.instruments
            if inst["name"] == asset and inst["instrument_type"] == "FUT"
        ]
        futures = sorted(futures, key=lambda x: x["expiry"])

        if not futures:
            return None

        # Liquidity filter
        liquid_futures = self._filter_liquid_futures(futures)
        if not liquid_futures:
            liquid_futures = futures  # fallback

        today = date.today()

        # --- Adaptive logic ---
        # Near expiry if within rollover window
        for fut in liquid_futures:
            if fut["expiry"] - today <= timedelta(days=self.rollover_days):
                continue  # skip expiring soon
            nearest = fut
            break
        else:
            nearest = liquid_futures[0]

        # High volatility → prefer far month
        if iv > 0.25 or prefer_far:
            return liquid_futures[-1]

        return nearest
