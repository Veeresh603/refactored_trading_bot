import math
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("OptionsOptimizer")

def _norm_cdf(x): 
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _bs_price(spot, strike, T, sigma, is_call):
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1 = (math.log(spot/strike) + 0.5*sigma*sigma*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if is_call:
        return spot * _norm_cdf(d1) - strike * _norm_cdf(d2)
    else:
        return strike * _norm_cdf(-d2) - spot * _norm_cdf(-d1)

class OptionsOptimizer:
    def __init__(self, cfg: dict, lot_size: int = 50):
        self.cfg = cfg
        self.lot_size = lot_size
        logger.info(f"⚙️ OptionsOptimizer initialized with lot_size={lot_size}")

    def evaluate(self, decision: dict, now=None):
        """
        Build an option contract around a decision.
        decision keys: asset, spot, strike_offset, expiry ('weekly'|'monthly'), signal in {-1,0,1}
        Returns dict: symbol, asset, strike, expiry(date), is_call, iv, premium
        """
        if now is None:
            now = datetime.now()

        spot = float(decision.get("spot", 0))
        if not spot:
            logger.warning("No spot price for optimizer; skipping.")
            return None

        is_call = True  # simple demo mapping; you can map signal to call/put if you like
        if decision.get("signal", 0) < 0:
            is_call = False

        # choose strike = rounded ATM + offset
        step = 100  # NIFTY step
        atm = round(spot / step) * step
        strike = int(atm + int(decision.get("strike_offset", 0)))

        # choose expiry
        expiry_kind = decision.get("expiry", "weekly")
        if expiry_kind == "weekly":
            expiry_date = (now + timedelta(days=7)).date()
        else:
            # 30 days out as a simple "monthly"
            expiry_date = (now + timedelta(days=30)).date()

        # IV guess
        iv = float(self.cfg.get("implied_vol", 0.20)) if "implied_vol" in self.cfg else 0.20

        # premium via BS
        T = max(1, (expiry_date - now.date()).days) / 365.0
        premium = _bs_price(spot, strike, T, iv, is_call)

        symbol = f"NIFTY{strike}{'CE' if is_call else 'PE'}"
        contract = {
            "symbol": symbol,
            "asset": "NIFTY",
            "strike": strike,
            "expiry": expiry_date,
            "is_call": is_call,
            "iv": iv,
            "premium": premium,  # ✅ used as price
        }
        logger.info(f"📦 Built contract: {contract}")
        return contract
