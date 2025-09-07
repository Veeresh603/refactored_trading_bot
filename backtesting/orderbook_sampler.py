# backtesting/orderbook_sampler.py
"""
Robust OrderBookSampler

Supports either:
 - OrderBookSampler(df=..., levels=..., ...)
 - OrderBookSampler(levels_list)  # convenience positional form widely used in tests

Behavior:
 - available_liquidity(idx, side=1) returns sum of asks (side>0) or bids (side<0) for levels,
   or volume * liquidity_fraction for df/list rows.
 - execute(...) returns dict with keys:
    executed, remaining, vwap, liquidity_used, latency_ms
 - VWAP is computed when `levels` snapshots are provided (consumes top-of-book levels).
 - Defensive handling for DataFrame-like vs list-of-dicts rows.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class OrderBookSampler:
    def __init__(
        self,
        df: Optional[Any] = None,
        levels: Optional[List[Dict[str, Any]]] = None,
        liquidity_fraction: float = 1.0,
        seed: int = 0,
        latency_mean_ms: float = 0.0,
        latency_jitter_ms: float = 0.0,
    ):
        """
        Constructor is forgiving: if user passes a single positional list argument
        (common test usage `OrderBookSampler(levels)`), treat it as `levels`.
        """
        # If caller passed only one positional arg intended as levels (e.g. OrderBookSampler(levels_list)),
        # Python will have bound that list to `df`. Detect that case:
        if levels is None and df is not None and isinstance(df, list):
            # detect likely level snapshot list: list of dicts with 'bids'/'asks' keys
            sample0 = df[0] if len(df) > 0 else None
            if isinstance(sample0, dict) and any(k in sample0 for k in ("bids", "asks")):
                levels = df
                df = None

        self.df = df
        self.levels = levels
        self.liquidity_fraction = float(liquidity_fraction)
        self.rng = np.random.RandomState(int(seed))
        self.last_latency_ms = 0.0
        self.latency_mean_ms = float(latency_mean_ms)
        self.latency_jitter_ms = float(latency_jitter_ms)

    def _row_for_idx(self, idx: int):
        """Return a dict-like row for the given idx, supporting both DataFrame and list-of-dicts."""
        if self.df is None:
            return None
        # DataFrame-like with .iloc
        if hasattr(self.df, "iloc"):
            try:
                if int(idx) >= len(self.df):
                    return None
                return self.df.iloc[int(idx)]
            except Exception:
                return None
        # list-like sequence (list of dicts)
        try:
            return self.df[int(idx)]
        except Exception:
            return None

    def available_liquidity(self, idx: int, side: int = 1) -> float:
        """
        Return available liquidity for the given idx and side.
        side > 0 -> asks (buy); side < 0 -> bids (sell).
        Default side=1 to be convenient for tests that only pass idx.
        """
        # If explicit level snapshots provided
        if self.levels is not None:
            if int(idx) >= len(self.levels):
                return 0.0
            book = self.levels[int(idx)]
            if not book:
                return 0.0
            if side > 0:
                return float(sum(sz for _, sz in book.get("asks", [])))
            elif side < 0:
                return float(sum(sz for _, sz in book.get("bids", [])))
            return 0.0

        # If time-series df/list is provided
        if self.df is not None:
            row = self._row_for_idx(idx)
            if row is None:
                return 0.0
            # row might be a pandas Series or a dict-like object; use .get if available
            try:
                vol = float(row.get("volume", 0.0))
            except Exception:
                try:
                    vol = float(row["volume"])
                except Exception:
                    vol = 0.0
            return float(vol) * float(self.liquidity_fraction)

        # default
        return 0.0

    def _vwap_from_levels(self, levels: List[Tuple[float, float]], requested: float) -> Tuple[Optional[float], float]:
        """
        Compute VWAP when consuming 'requested' units from levels (list of (price, size)).
        Returns (vwap_or_None, executed).
        """
        req = float(requested)
        if req <= 0.0:
            return None, 0.0
        rem = req
        executed = 0.0
        cost = 0.0
        for price, size in levels:
            if rem <= 0.0:
                break
            take = min(rem, float(size))
            cost += float(price) * float(take)
            executed += float(take)
            rem -= take
        if executed <= 0.0:
            return None, 0.0
        return float(cost / executed), float(executed)

    def _simulate_latency_ms(self) -> float:
        """Return a simulated latency in milliseconds and store last_latency_ms."""
        if self.latency_jitter_ms > 0.0:
            jitter = float(self.rng.normal(self.latency_mean_ms, self.latency_jitter_ms))
        else:
            jitter = float(self.latency_mean_ms)
        lat = float(max(0.0, jitter))
        self.last_latency_ms = lat
        return lat

    def execute(self, idx: int, side: int, requested_units: float) -> Dict[str, Any]:
        """
        Execute requested_units at index idx and return dict with executed, remaining, vwap, liquidity_used, latency_ms

        - When levels are available, compute realistic VWAP by consuming levels top-down.
        - When df is available, executed = min(requested, volume * liquidity_fraction)
          and vwap uses 'close' if present (best-effort).
        """
        latency_ms = self._simulate_latency_ms()
        requested_abs = float(abs(requested_units))
        if requested_abs <= 0.0:
            return {"executed": 0.0, "remaining": 0.0, "vwap": None, "liquidity_used": 0.0, "latency_ms": float(latency_ms)}

        # available liquidity in this bar for the given side
        avail = float(self.available_liquidity(idx, side=side))
        executed = float(min(requested_abs, avail))
        remaining = float(max(0.0, requested_abs - executed))
        liquidity_used = float(executed)
        vwap = None

        # Prefer levels (orderbook snapshot) if present
        if self.levels is not None and int(idx) < len(self.levels):
            book = self.levels[int(idx)]
            levels = book.get("asks" if side > 0 else "bids", [])
            if levels:
                vwap_val, exec_from_levels = self._vwap_from_levels(levels, executed)
                if exec_from_levels > 0.0:
                    vwap = float(vwap_val) if vwap_val is not None else None
                    executed = float(exec_from_levels)
                    liquidity_used = float(executed)
                    remaining = float(max(0.0, requested_abs - executed))
                else:
                    executed = 0.0
                    liquidity_used = 0.0
                    remaining = float(requested_abs)

        # If levels not present, attempt df-based vwap/price
        elif self.df is not None:
            row = self._row_for_idx(idx)
            if row is not None:
                try:
                    v = row.get("close", None)
                    vwap = float(v) if v is not None else None
                except Exception:
                    try:
                        vwap = float(row["close"])
                    except Exception:
                        vwap = None

        # ensure numeric types
        executed = float(executed)
        remaining = float(remaining)
        liquidity_used = float(liquidity_used)
        latency_ms = float(latency_ms)
        vwap_out = float(vwap) if vwap is not None else None

        return {
            "executed": executed,
            "remaining": remaining,
            "vwap": vwap_out,
            "liquidity_used": liquidity_used,
            "latency_ms": latency_ms,
        }
