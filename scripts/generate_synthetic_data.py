# scripts/generate_synthetic_data.py
"""
Generate a synthetic historical CSV at data/historical.csv.

Columns:
 - time (ISO timestamp)
 - open, high, low, close (floats)
 - volume (int)

Usage:
    python scripts/generate_synthetic_data.py --out data/historical.csv --days 10 --freq 1T
"""
from __future__ import annotations
import os
import argparse
from datetime import datetime, timedelta
import math
import random

import numpy as np
import pandas as pd


def generate_gbm_path(S0: float, mu: float, sigma: float, steps: int, dt: float):
    """Geometric Brownian Motion path"""
    prices = [S0]
    for _ in range(steps - 1):
        z = np.random.normal()
        S_prev = prices[-1]
        S_next = S_prev * math.exp((mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z)
        prices.append(S_next)
    return np.array(prices)


def make_ohlcv(prices: np.ndarray, vol_scale: float = 1.0):
    """Create OHLCV from tick prices (simple approach)"""
    opens = prices.copy()
    closes = prices.copy()
    highs = np.maximum(opens, closes) * (1 + np.random.rand(len(prices)) * 0.002)
    lows = np.minimum(opens, closes) * (1 - np.random.rand(len(prices)) * 0.002)
    volumes = (np.abs(np.random.randn(len(prices))) * 1000 * vol_scale).astype(int) + 1
    return opens, highs, lows, closes, volumes


def build_dataframe(start: datetime, periods: int, freq_minutes: int, S0: float = 20000.0):
    dt_days = freq_minutes / (60 * 24)  # fraction of day
    mu = 0.0     # drift
    sigma = 0.2  # daily vol ~20% annualized (approx)
    prices = generate_gbm_path(S0=S0, mu=mu, sigma=sigma, steps=periods, dt=dt_days)
    opens, highs, lows, closes, volumes = make_ohlcv(prices)
    times = [start + timedelta(minutes=freq_minutes * i) for i in range(periods)]
    df = pd.DataFrame({
        "time": [t.isoformat(sep=" ") for t in times],
        "open": np.round(opens, 6),
        "high": np.round(highs, 6),
        "low": np.round(lows, 6),
        "close": np.round(closes, 6),
        "volume": volumes,
    })
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/historical.csv", help="Output CSV path")
    p.add_argument("--days", type=float, default=5.0, help="How many days to simulate (market 24/7 here)")
    p.add_argument("--freq", default="1T", help="Frequency: e.g. '1T' (1 minute), '5T'")
    p.add_argument("--start", default=None, help="ISO start time, e.g. '2025-01-01 09:15:00' (defaults to now - days)")
    p.add_argument("--seed", type=int, default=123, help="Random seed")
    args = p.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # parse freq to minutes (simple)
    if args.freq.endswith("T"):
        freq_minutes = int(args.freq[:-1])
    else:
        # fallback to minutes
        freq_minutes = 1

    periods = int((24 * 60 / freq_minutes) * args.days)
    if periods < 10:
        periods = max(10, periods)

    if args.start:
        start = datetime.fromisoformat(args.start)
    else:
        start = datetime.utcnow() - timedelta(days=args.days)

    df = build_dataframe(start=start, periods=periods, freq_minutes=freq_minutes, S0=20000.0)

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(args.out, index=False)
    print(f"Generated synthetic data -> {args.out} ({len(df)} rows, start={df['time'].iloc[0]})")


if __name__ == "__main__":
    main()
