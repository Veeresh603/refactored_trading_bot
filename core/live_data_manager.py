"""
Live Data Manager
-----------------
- Manages real-time market data feed
- Supports broker adapters (Zerodha, AngelOne, etc.)
- Auto-reconnect on errors
- Caches last known tick
"""

import logging
import time
import random

logger = logging.getLogger("LiveDataManager")


class LiveDataManager:
    def __init__(self, instruments, broker="zerodha", reconnect_interval=5):
        """
        Args:
            instruments (list): list of instrument tokens
            broker (str): broker name (zerodha | angel | paper)
            reconnect_interval (int): seconds before retry on failure
        """
        self.instruments = instruments
        self.broker = broker
        self.reconnect_interval = reconnect_interval
        self.last_ticks = {inst: None for inst in instruments}
        self.connected = False

        logger.info(f"📡 LiveDataManager initialized for broker={broker}, instruments={instruments}")
        self._connect()

    def _connect(self):
        """
        Connect to broker feed.
        """
        try:
            # TODO: replace with actual broker SDK
            logger.info(f"🔌 Connecting to {self.broker} data feed...")
            time.sleep(1)
            self.connected = True
            logger.info("✅ Live data feed connected")
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            self.connected = False

    def _simulate_tick(self, inst):
        """
        Simulated tick generator (for testing without broker).
        """
        return {"instrument": inst, "last_price": random.uniform(19500, 20500), "timestamp": time.time()}

    def get_latest(self, instrument):
        """
        Get latest tick for a given instrument.
        Returns cached tick if live feed fails.
        """
        try:
            if not self.connected:
                logger.warning("⚠️ Feed not connected, retrying...")
                self._connect()
                if not self.connected:
                    return self.last_ticks[instrument]

            # --- Replace this with actual broker tick fetch ---
            tick = self._simulate_tick(instrument)

            if tick:
                self.last_ticks[instrument] = tick
                return tick
            else:
                logger.warning(f"⚠️ No new tick for {instrument}, using last known")
                return self.last_ticks[instrument]

        except Exception as e:
            logger.error(f"❌ Error fetching tick for {instrument}: {e}")
            time.sleep(self.reconnect_interval)
            return self.last_ticks[instrument]

    def get_all_latest(self):
        """
        Get ticks for all instruments.
        """
        return {inst: self.get_latest(inst) for inst in self.instruments}
