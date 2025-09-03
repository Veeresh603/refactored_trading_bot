"""
Data Manager
------------
- Supports live streaming (Zerodha Kite)
- Stores tick data internally
- Provides latest ticks + history
- Can also load historical OHLCV from CSV (for backtest/training)
"""

import os
import pandas as pd
import logging
from datetime import datetime
from kiteconnect import KiteTicker


class DataManager:
    def __init__(self, kite=None, data_dir="data"):
        self.kite = kite
        self.data = {}
        self.data_dir = data_dir
        self.logger = logging.getLogger("DataManager")

    # -----------------------------
    # Live Streaming (Zerodha Kite)
    # -----------------------------
    def start_streaming(self, api_key, access_token, tokens, on_tick_callback=None):
        """
        Start streaming ticks.

        Args:
            api_key (str): Kite API key
            access_token (str): Kite access token
            tokens (list[int]): List of instrument tokens
            on_tick_callback (callable): Optional callback for ticks
        """
        kws = KiteTicker(api_key, access_token)

        def on_ticks(ws, ticks):
            for token in set([t["instrument_token"] for t in ticks]):
                self.data[token] = [t for t in ticks if t["instrument_token"] == token]
            if on_tick_callback:
                on_tick_callback(ticks)

        def on_connect(ws, response):
            self.logger.info(f"✅ Connected to Kite stream, subscribing: {tokens}")
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)

        def on_close(ws, code, reason):
            self.logger.warning(f"⚠️ Kite stream closed: {reason}")

        kws.on_ticks = on_ticks
        kws.on_connect = on_connect
        kws.on_close = on_close
        kws.connect(threaded=True)

    def get_latest(self, token):
        """Get latest tick for a given instrument token."""
        return self.data.get(token, [{}])[-1]

    # -----------------------------
    # Historical CSV Loader
    # -----------------------------
    def load_csv(self, symbol, filename=None):
        """
        Load historical OHLCV data from CSV.

        Args:
            symbol (str): e.g., "NIFTY"
            filename (str): optional CSV path

        Returns:
            pd.DataFrame
        """
        path = filename or os.path.join(self.data_dir, f"{symbol}.csv")
        if not os.path.exists(path):
            self.logger.error(f"❌ CSV not found: {path}")
            return pd.DataFrame()

        df = pd.read_csv(path, parse_dates=["time"])
        df = df.sort_values("time").reset_index(drop=True)
        return df
