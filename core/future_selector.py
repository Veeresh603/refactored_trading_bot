"""
Future Selector
---------------
- Selects correct futures contract for a given asset
- Uses nearest expiry available
"""

from datetime import date


class FutureSelector:
    def __init__(self, broker):
        self.broker = broker
        self.instruments = broker.kite.instruments("NFO")

    def get_nearest_future(self, asset):
        """Find nearest FUT contract for asset"""
        futures = [inst for inst in self.instruments if inst["name"] == asset and inst["instrument_type"] == "FUT"]
        expiries = sorted({f["expiry"] for f in futures})
        today = date.today()
        for e in expiries:
            if e >= today:
                fut = [f for f in futures if f["expiry"] == e]
                if fut:
                    return fut[0]["tradingsymbol"], fut[0]["instrument_token"]
        return None, None
