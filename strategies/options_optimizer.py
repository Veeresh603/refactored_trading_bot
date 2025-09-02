import logging
from datetime import datetime, timedelta

logger = logging.getLogger("OptionsOptimizer")

class OptionsOptimizer:
    def __init__(self, config, lot_size=50):
        self.enabled = config["enabled"]
        self.strike_mode = config["strike_selection"]
        self.expiry_mode = config["expiry_selection"]
        self.max_open_positions = config["max_open_positions"]
        self.lot_size = lot_size

        logger.info(f"⚙️ Options Optimizer enabled={self.enabled}, "
                    f"strike={self.strike_mode}, expiry={self.expiry_mode}, "
                    f"max_open_positions={self.max_open_positions}")

    def get_strike(self, spot_price, iv=None, greeks=None):
        """Choose strike price based on mode, IV and Greeks"""
        if self.strike_mode == "ATM":
            return round(spot_price / 100) * 100
        elif self.strike_mode == "ITM":
            return round(spot_price / 100) * 100 - 100
        elif self.strike_mode == "OTM":
            return round(spot_price / 100) * 100 + 100
        elif self.strike_mode == "dynamic":
            # IV/Greeks based logic
            atm = round(spot_price / 100) * 100
            if iv is None:
                iv = 0.2
            if greeks is None:
                greeks = {"delta": 0, "gamma": 0, "vega": 0}

            # If portfolio delta too positive → choose slightly ITM Put
            if greeks["delta"] > 50:
                return atm - 200
            # If vega risk high and IV high → prefer selling OTM Calls
            elif greeks["vega"] > 1000 and iv > 0.25:
                return atm + 200
            # Otherwise default to ATM
            else:
                return atm
        return round(spot_price / 100) * 100

    def get_expiry(self, today=None, iv=None):
        """Choose expiry date dynamically"""
        if today is None:
            today = datetime.now().date()

        if self.expiry_mode == "weekly":
            # Next Thursday
            days_ahead = (3 - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            expiry = today + timedelta(days=days_ahead)
        elif self.expiry_mode == "monthly":
            # Last Thursday of the month
            next_month = today.month + 1 if today.month < 12 else 1
            year = today.year if today.month < 12 else today.year + 1
            expiry = datetime(year, next_month, 1).date() - timedelta(days=1)
            while expiry.weekday() != 3:
                expiry -= timedelta(days=1)
        elif self.expiry_mode == "dynamic":
            # IV-based expiry selection
            if iv and iv > 0.3:
                # High IV → shorter expiry
                days_ahead = (3 - today.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                expiry = today + timedelta(days=days_ahead)
            else:
                # Low IV → go monthly
                next_month = today.month + 1 if today.month < 12 else 1
                year = today.year if today.month < 12 else today.year + 1
                expiry = datetime(year, next_month, 1).date() - timedelta(days=1)
                while expiry.weekday() != 3:
                    expiry -= timedelta(days=1)
        else:
            # Default weekly
            days_ahead = (3 - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            expiry = today + timedelta(days=days_ahead)

        return expiry

    def build_option_contract(self, asset, spot_price, is_call=True, iv=None, greeks=None):
        """Return an option contract dict"""
        if not self.enabled:
            logger.warning("Options Optimizer disabled, returning ATM weekly CE")
            strike = round(spot_price / 100) * 100
            expiry = self.get_expiry()
        else:
            strike = self.get_strike(spot_price, iv, greeks)
            expiry = self.get_expiry(iv=iv)

        option_symbol = f"{asset}_{expiry.strftime('%Y%m%d')}_{strike}{'CE' if is_call else 'PE'}"

        return {
            "symbol": option_symbol,
            "strike": strike,
            "expiry": expiry,
            "is_call": is_call,
            "qty": self.lot_size,
        }
