import os
import threading
from kiteconnect import KiteConnect, KiteTicker
from dotenv import load_dotenv

load_dotenv()


class LiveDataManager:
    def __init__(self, api_key=None, access_token=None, instruments=None):
        self.api_key = api_key or os.getenv("KITE_API_KEY")
        self.access_token = access_token or os.getenv("KITE_ACCESS_TOKEN")
        self.instruments = instruments or [256265]  # Default: NIFTY index

        self.kite = KiteConnect(api_key=self.api_key)
        self.kite.set_access_token(self.access_token)

        self.kws = KiteTicker(self.api_key, self.access_token)

        self.latest_ticks = {}
        self.lock = threading.Lock()

        # Register callbacks
        self.kws.on_ticks = self.on_ticks
        self.kws.on_connect = self.on_connect
        self.kws.on_close = self.on_close

        # Run in a separate thread
        self.thread = threading.Thread(target=self.kws.connect, kwargs={"threaded": True})
        self.thread.daemon = True
        self.thread.start()

    def on_ticks(self, ws, ticks):
        with self.lock:
            for tick in ticks:
                self.latest_ticks[tick["instrument_token"]] = tick

    def on_connect(self, ws, response):
        ws.subscribe(self.instruments)
        ws.set_mode(ws.MODE_FULL, self.instruments)

    def on_close(self, ws, code, reason):
        print("⚠️ LiveDataManager: WebSocket closed:", code, reason)

    def get_latest(self, instrument_token=None):
        """Get the most recent tick data for an instrument."""
        with self.lock:
            if instrument_token:
                return self.latest_ticks.get(instrument_token, None)
            return self.latest_ticks.copy()
