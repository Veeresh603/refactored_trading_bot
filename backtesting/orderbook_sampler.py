# backtesting/orderbook_sampler.py
"""
AdvancedOrderBookSampler: stochastic liquidity model, latency jitter, VWAP partial fills.

API (same as before):
- OrderBookSampler(df=None, liquidity_fraction=0.1, seed=None, latency_mean=0, latency_jitter=0)
- available_liquidity(idx, side=1)
- execute(idx, side, requested_units) -> {"executed": float, "vwap": float}

This implementation:
- Uses df['volume'] if available, otherwise default_liq.
- Simulates available liquidity as Draw from: max(0, Normal(volume * liquidity_fraction, sigma))
- latency jitter is returned via `last_latency_ms` attribute (useful if backtester integrates latency)
"""
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

class OrderBookSampler:
    def __init__(self, df: Optional[pd.DataFrame] = None, liquidity_fraction: float = 0.1, default_liq: float = 1.0, seed: Optional[int] = None, latency_mean_ms: float = 0.0, latency_jitter_ms: float = 0.0):
        self.df = df
        self.liquidity_fraction = float(liquidity_fraction)
        self.default_liq = float(default_liq)
        self.rng = np.random.RandomState(seed)
        self.latency_mean_ms = float(latency_mean_ms)
        self.latency_jitter_ms = float(latency_jitter_ms)
        self.last_latency_ms = 0.0

    def _sample_available(self, vol: float) -> float:
        base = max(0.0, vol * self.liquidity_fraction)
        # volatility of liquidity: 10% of base or min 0.1
        sigma = max(0.1 * base, 0.01)
        sampled = self.rng.normal(loc=base, scale=sigma)
        return max(0.0, float(sampled))

    def available_liquidity(self, idx: int, side: int = 1) -> float:
        if self.df is None:
            return float(self.default_liq)
        try:
            vol = float(self.df.iloc[int(idx)].get("volume", 0.0))
            if vol <= 0.0:
                return float(self.default_liq)
            return float(self._sample_available(vol))
        except Exception:
            return float(self.default_liq)

    def execute(self, idx: int, side: int, requested_units: float) -> Dict[str, Any]:
        # simulate latency jitter
        jitter = self.rng.normal(self.latency_mean_ms, self.latency_jitter_ms) if self.latency_jitter_ms > 0 else self.latency_mean_ms
        self.last_latency_ms = max(0.0, float(jitter))

        avail = self.available_liquidity(idx, side=side)
        executed = float(min(abs(requested_units), avail))

        # vwap approx: if df present, use close +/- random micro slippage scaled by volatility
        vwap = None
        if self.df is not None:
            price = float(self.df.iloc[int(idx)].get("close", 0.0))
            high = float(self.df.iloc[int(idx)].get("high", price))
            low  = float(self.df.iloc[int(idx)].get("low", price))
            # micro jitter scaled by bar range
            bar_range = max(1e-6, high - low)
            micro = self.rng.normal(0.0, bar_range * 0.01)
            vwap = float(price + micro)
        return {"executed": executed, "vwap": vwap}
