"""
Data Downloader with Option Chain
---------------------------------
- Fetches historical OHLCV (spot)
- Fetches option chain & Greeks (requires Zerodha API or NSE API)
"""

import pandas as pd
from datetime import datetime, timedelta
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega


class DataDownloader:
    def __init__(self, broker):
        self.broker = broker

    def download_ohlcv(self, instrument_token, days=180, interval="15minute"):
        """Download last `days` of OHLCV data for spot index/stock"""
        end = datetime.now()
        start = end - timedelta(days=days)

        data = self.broker.kite.historical_data(
            instrument_token,
            from_date=start.strftime("%Y-%m-%d"),
            to_date=end.strftime("%Y-%m-%d"),
            interval=interval,
            continuous=False
        )

        df = pd.DataFrame(data)
        if not df.empty:
            df.rename(columns={"date": "timestamp"}, inplace=True)
            df.set_index("timestamp", inplace=True)
        return df

    def compute_greeks_from_chain(self, option_chain, spot_price, r=0.05, T=0.05):
        """
        Compute portfolio-level Greeks from option chain snapshot
        Requires py_vollib for Black-Scholes Greeks
        """
        total_delta, total_gamma, total_vega = 0, 0, 0
        for opt in option_chain:
            try:
                S = spot_price
                K = opt["strike"]
                sigma = opt.get("iv", 0.2)  # implied volatility
                is_call = opt["instrument_type"] == "CE"

                d = delta("c" if is_call else "p", S, K, T, r, sigma)
                g = gamma("c", S, K, T, r, sigma)
                v = vega("c", S, K, T, r, sigma)

                qty = opt.get("open_interest", 1)

                total_delta += d * qty
                total_gamma += g * qty
                total_vega += v * qty
            except Exception:
                continue

        return total_delta, total_gamma, total_vega
