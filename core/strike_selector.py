"""
Strike Selector with Zerodha Kite Option Chain
----------------------------------------------
- Dynamically selects ATM/ITM/OTM strikes
- Uses volume + OI filters for liquidity
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

        # Cache all NFO instruments
        self.instruments = broker.kite.instruments("NFO")

    def get_nearest_expiry(self, asset):
        """Pick nearest expiry available for the given asset"""
        expiries = sorted({inst["expiry"] for inst in self.instruments if inst["name"] == asset})
        today = date.today()
        for e in expiries:
            if e >= today:
                return e.strftime("%Y-%m-%d")
        return expiries[0].strftime("%Y-%m-%d")

    def get_strike(self, spot_price, signal, step=50):
        atm = round(spot_price / step) * step
        if signal == "BUY":
            return atm + step
        elif signal == "SELL":
            return atm - step
        else:
            return atm

    def _fetch_option_chain(self, asset, expiry):
        """Filter NFO instruments for given asset + expiry"""
        option_chain = [
            inst for inst in self.instruments
            if inst["name"] == asset and str(inst["expiry"]) == expiry
        ]
        return option_chain

    def select_option(self, asset, spot_price, signal):
        """Select the most liquid option contract"""
        expiry = self.get_nearest_expiry(asset)
        strike = self.get_strike(spot_price, signal)
        option_type = "CE" if signal == "BUY" else "PE"

        # Get contracts for this asset + expiry
        option_chain = self._fetch_option_chain(asset, expiry)

        # Candidate strikes around ATM ± 2 steps
        candidate_strikes = [strike - 100, strike - 50, strike, strike + 50, strike + 100]

        # Filter contracts
        candidates = [
            o for o in option_chain
            if o["strike"] in candidate_strikes and o["instrument_type"] == option_type
        ]

        if not candidates:
            # fallback ATM
            return f"{asset}{expiry}{strike}{option_type}"

        # Fetch LTP, OI, Volume for candidates
        tokens = [c["instrument_token"] for c in candidates]
        ltp_data = self.broker.kite.ltp(tokens)

        liquid = []
        for c in candidates:
            token = c["instrument_token"]
            info = ltp_data.get(str(token), {})
            oi = info.get("oi", 0)
            volume = info.get("volume", 0)
            if volume >= self.min_volume and oi >= self.min_oi:
                liquid.append((c, volume, oi))

        if not liquid:
            # fallback ATM CE/PE
            return candidates[0]["tradingsymbol"]

        # Pick most liquid by volume*OI
        best = max(liquid, key=lambda x: x[1] * x[2])
        return best[0]["tradingsymbol"]
