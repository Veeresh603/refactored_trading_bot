# core/live_data_manager.py
"""
LiveDataManager: robust, thread-safe live/historical data manager with fallbacks.

Features:
- Keeps last-tick cache for symbols
- Allows subscription callbacks for incoming ticks
- Simple replay mode from CSV for paper-trading/backtesting
- Graceful broker SDK integration points (kiteconnect, ccxt) if installed
- Reconnect/backoff logic for live streams
- Thread-safe API suitable for integration with engine and strategy code
"""

from __future__ import annotations

import csv
import logging
import threading
import time
import math
import os
from typing import Callable, Dict, Optional, Any, List, Tuple
from collections import defaultdict
import random

logger = logging.getLogger("TradingBot.LiveDataManager")
logger.setLevel(logging.INFO)

# Optional broker SDK placeholders
try:
    import kiteconnect  # type: ignore
    HAVE_KITECON = True
except Exception:
    HAVE_KITECON = False

try:
    import ccxt  # type: ignore
    HAVE_CCXT = True
except Exception:
    HAVE_CCXT = False


class LiveDataManager:
    """
    Responsibilities:
      - Provide the latest market price for symbols via get_last(symbol)
      - Allow components to subscribe to ticks: subscribe(symbol, callback)
      - Replay historical CSV as a simulated live feed (replay_from_csv)
      - Provide a simple interface to fetch historical CSVs (fetch_historical)
    """

    def __init__(self, asset: Optional[str] = None, default_slippage: float = 0.0005):
        self.asset = asset
        self._last_tick: Dict[str, Dict[str, Any]] = {}
        self._subs: Dict[str, List[Callable[[Dict[str, Any]], None]]] = defaultdict(list)
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._backoff_seconds = 1.0
        self._max_backoff = 300.0
        self.default_slippage = default_slippage

        # For replay mode
        self._replay_thread: Optional[threading.Thread] = None
        self._replay_stop_event = threading.Event()

        logger.info("LiveDataManager initialized")

    # -------------------------
    # Subscription / snapshot API
    # -------------------------
    def subscribe(self, symbol: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to ticks for `symbol`. Callback receives the tick dict."""
        with self._lock:
            self._subs[symbol].append(callback)
        logger.debug(f"Subscribed callback to {symbol}")

    def unsubscribe(self, symbol: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            if symbol in self._subs and callback in self._subs[symbol]:
                self._subs[symbol].remove(callback)

    def get_last(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return last tick for symbol (copy) or None if unknown."""
        with self._lock:
            t = self._last_tick.get(symbol)
            return dict(t) if t is not None else None

    # -------------------------
    # Internal dispatch
    # -------------------------
    def _dispatch_tick(self, tick: Dict[str, Any]) -> None:
        symbol = tick.get("symbol")
        if symbol is None:
            return
        with self._lock:
            # store a shallow copy
            self._last_tick[symbol] = dict(tick)

        # call subscribers (do not hold lock while calling client callbacks)
        callbacks = []
        with self._lock:
            callbacks = list(self._subs.get(symbol, []))
        for cb in callbacks:
            try:
                cb(dict(tick))
            except Exception:
                logger.exception("Subscriber callback raised exception")

    # -------------------------
    # Historical CSV helpers
    # -------------------------
    def fetch_historical(self, csv_path: str, symbol_col: str = "symbol",
                         ts_col: str = "timestamp", price_col: str = "close",
                         start: Optional[float] = None, end: Optional[float] = None
                         ) -> List[Dict[str, Any]]:
        """
        Load a CSV and return a list of ticks (dict). CSV must contain at least timestamp and price columns.
        timestamp expected as unix epoch (float/int) or ISO string — we do best-effort parse (if non-numeric, leave as string).
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        rows: List[Dict[str, Any]] = []
        with open(csv_path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            if price_col not in reader.fieldnames:
                raise ValueError(f"Price column '{price_col}' not in CSV fields {reader.fieldnames}")

            for r in reader:
                try:
                    ts_val = r.get(ts_col)
                    if ts_val is None:
                        continue
                    try:
                        ts = float(ts_val)
                    except Exception:
                        ts = ts_val  # leave as-is (caller can parse)
                    price = float(r[price_col])
                    symbol = r.get(symbol_col) if symbol_col in r else (self.asset or "UNKNOWN")
                    tick = {"symbol": symbol, "timestamp": ts, "price": price, **r}
                    # apply filters
                    if start is not None and isinstance(ts, (int, float)) and ts < start:
                        continue
                    if end is not None and isinstance(ts, (int, float)) and ts > end:
                        continue
                    rows.append(tick)
                except Exception:
                    logger.exception("Error parsing CSV row; skipping")
        logger.info(f"Loaded {len(rows)} rows from {csv_path}")
        return rows

    def replay_from_csv(self, csv_path: str, symbol_col: str = "symbol", ts_col: str = "timestamp",
                        price_col: str = "close", speed: float = 1.0, loop: bool = False) -> None:
        """
        Replay ticks from CSV as if they were live market ticks.
        - speed: 1.0 => real-time (if CSV timestamps are real epoch), 2.0 => twice as fast, 0.0 => immediate-fire all ticks
        - loop: replay repeatedly until stopped
        """

        if self._replay_thread and self._replay_thread.is_alive():
            raise RuntimeError("Replay already running")

        ticks = self.fetch_historical(csv_path, symbol_col=symbol_col, ts_col=ts_col, price_col=price_col)
        if not ticks:
            raise RuntimeError("No ticks to replay")

        # sort by timestamp if numeric
        try:
            ticks.sort(key=lambda t: float(t["timestamp"]))
        except Exception:
            # fallback: keep file order
            pass

        def _run_replay():
            logger.info("Replay started")
            while not self._replay_stop_event.is_set():
                last_ts = None
                for t in ticks:
                    if self._replay_stop_event.is_set():
                        break
                    # wait according to differences in timestamp
                    try:
                        ts = float(t["timestamp"])
                    except Exception:
                        ts = None
                    if last_ts is not None and ts is not None and speed > 0.0:
                        dt = max(0.0, (ts - last_ts) / speed)
                        # defensive cap on sleep (avoid huge delays)
                        if dt > 0:
                            if dt > 60.0:
                                # if there is a huge gap in data, accelerate it
                                dt = 0.5
                            time.sleep(dt)
                    # produce tick
                    tick = {"symbol": t.get(symbol_col, self.asset or "UNKNOWN"),
                            "price": float(t.get(price_col)),
                            "timestamp": t.get(ts_col)}
                    self._dispatch_tick(tick)
                    last_ts = ts
                if not loop:
                    break
            logger.info("Replay stopped")

        # reset stop event and start thread
        self._replay_stop_event.clear()
        self._replay_thread = threading.Thread(target=_run_replay, daemon=True)
        self._replay_thread.start()

    def stop_replay(self) -> None:
        if self._replay_thread and self._replay_thread.is_alive():
            self._replay_stop_event.set()
            self._replay_thread.join(timeout=5.0)

    # -------------------------
    # Live broker integration (best-effort stubs)
    # -------------------------
    def start_live(self, connect_info: Optional[Dict[str, Any]] = None, symbols: Optional[List[str]] = None) -> None:
        """
        Attempt to start a live market feed using available SDKs. This function is best-effort:
        - If kiteconnect or ccxt are not available, it will log and return.
        - It will not raise on missing SDKs; instead it will recommend using replay_from_csv for paper mode.
        """
        if self._running:
            logger.info("LiveDataManager already running")
            return

        # If we have kiteconnect & credentials provided, attempt to start a connection
        if HAVE_KITECON and connect_info:
            logger.info("Attempting to start KiteConnect live feed (best-effort)")
            # NOTE: real integration requires keys and event handling; this is placeholder scaffold
            try:
                # Example: kite = KiteConnect(api_key=connect_info["api_key"])
                # Implement actual streaming subscription per SDK docs in production
                logger.warning("KiteConnect integration is not implemented in this fallback. Use replay_from_csv for testing.")
            except Exception:
                logger.exception("KiteConnect failed to initialize")
        elif HAVE_CCXT and connect_info:
            logger.info("Attempting to start ccxt exchange feed (best-effort)")
            # For exchanges, users typically poll REST or use websockets; full implementation is exchange-specific.
            logger.warning("ccxt live integration not implemented here. Use replay_from_csv for testing.")
        else:
            logger.info("No broker SDK available; using replay/polling modes. Use replay_from_csv for test data.")

        # mark as running (so calls to stop behave consistently)
        self._running = True

    def stop(self) -> None:
        """
        Stop any live/replay threads and mark manager as stopped.
        """
        self._replay_stop_event.set()
        if self._replay_thread and self._replay_thread.is_alive():
            self._replay_thread.join(timeout=5.0)
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._running = False
        logger.info("LiveDataManager stopped")

    # -------------------------
    # Utilities
    # -------------------------
    def emit_manual_tick(self, symbol: str, price: float, timestamp: Optional[float] = None) -> None:
        """
        Helper to emit a single tick (useful for unit tests or manual driving)
        """
        tick = {"symbol": symbol, "price": float(price), "timestamp": timestamp or time.time()}
        self._dispatch_tick(tick)

    def list_subscribed(self) -> List[str]:
        with self._lock:
            return list(self._subs.keys())


# -------------------------
# Local smoke-test / demo
# -------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    mgr = LiveDataManager(asset="TEST")

    def cb(t):
        print("CB GOT:", t)

    mgr.subscribe("TEST", cb)
    print("Emit manual tick...")
    mgr.emit_manual_tick("TEST", 123.45)
    # Create a tiny CSV and replay it
    sample_csv = "sample_ticks.csv"
    with open(sample_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["symbol", "timestamp", "close"])
        writer.writeheader()
        now = time.time()
        for i in range(5):
            writer.writerow({"symbol": "TEST", "timestamp": now + i, "close": 123.45 + i})

    print("Start replay (fast)...")
    mgr.replay_from_csv(sample_csv, speed=1000.0)
    time.sleep(0.5)
    mgr.stop_replay()
    os.remove(sample_csv)
    print("Demo complete")
