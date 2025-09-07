# cli/generate_features.py
"""
Generate supervised features + labels from a CSV of price bars.

Produces:
 - <out>/features.npy      : (n_samples, n_features) float32
 - <out>/labels.npy        : (n_samples,) int8 (or float32) labels (0/1/-1 as configured)
 - <out>/features_meta.json: metadata describing transformation (lookback, horizon, columns, seed)
 - <out>/prices.csv        : cleaned price series CSV used to compute features

Design goals:
 - deterministic, reproducible transforms
 - traceable metadata written to disk for training reproducibility
 - simple but practical feature set: returns, lag returns, rolling mean/std, volume
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import json
import logging
from typing import Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("TradingBot.GenerateFeatures")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")


def resolve_path(p: str) -> str | None:
    """Try to resolve a path from common locations."""
    if not p:
        return None
    cand = Path(p)
    if cand.exists():
        return str(cand)
    # some tolerant alternatives
    alt = Path.cwd() / p
    if alt.exists():
        return str(alt)
    return None


def load_csv_prices(csv_path: str) -> pd.DataFrame:
    """Load CSV file and validate expected columns. Returns DataFrame indexed by ascending time."""
    df = pd.read_csv(csv_path, parse_dates=["time"], infer_datetime_format=True)
    # required columns: time, open, high, low, close, volume
    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")
    # sort by time ascending, reset index
    df = df.sort_values("time").reset_index(drop=True)
    # basic cleaning: forward-fill NA on prices, volume fill zeros
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()
    df["volume"] = df["volume"].fillna(0.0)
    return df


def build_features_and_labels(
    prices: pd.DataFrame,
    lookback: int = 10,
    horizon: int = 1,
    label_mode: str = "direction",
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Build features and labels from price DataFrame.

    features: for each time t we produce features computed from t-lookback+1 .. t (inclusive)
    labels: computed at t+horizon (default 1) - direction: 1 if future return > 0, 0 otherwise.
    Returns (X, y, meta)
    """
    close = prices["close"].astype(float)
    volume = prices["volume"].astype(float)

    n = len(close)
    if n < lookback + horizon:
        raise ValueError("Not enough bars for requested lookback + horizon")

    # compute simple returns
    returns = close.pct_change().fillna(0.0)

    # precompute rolling stats on close
    roll_mean = close.rolling(window=lookback, min_periods=1).mean()
    roll_std = close.rolling(window=lookback, min_periods=1).std().fillna(0.0)

    # build lag features and rolling features at time t (uses up to t)
    feature_list = []
    feature_names = []

    # recent returns lags (1..lookback)
    for lag in range(1, lookback + 1):
        f = returns.shift(lag).fillna(0.0)
        feature_list.append(f)
        feature_names.append(f"ret_lag_{lag}")

    # rolling mean and std
    feature_list.append(roll_mean)
    feature_names.append("roll_mean")
    feature_list.append(roll_std)
    feature_names.append("roll_std")

    # volume and volume lags
    feature_list.append(volume)
    feature_names.append("volume")
    for lag in range(1, min(4, lookback) + 1):
        feature_list.append(volume.shift(lag).fillna(0.0))
        feature_names.append(f"vol_lag_{lag}")

    # Stack features into DataFrame aligned by index
    feats_df = pd.concat(feature_list, axis=1)
    feats_df.columns = feature_names

    # Truncate so that for index t we can compute label at t+horizon
    max_idx = n - horizon
    usable_idx = np.arange(lookback, max_idx)  # start at lookback so we have full lag history
    X = feats_df.iloc[usable_idx].to_numpy(dtype=np.float32)

    # Labels
    future_close = close.shift(-horizon)
    future_return = (future_close - close) / close
    if label_mode == "direction":
        y = (future_return.iloc[usable_idx] > 0.0).astype(np.int8).to_numpy()
    else:
        y = future_return.iloc[usable_idx].to_numpy(dtype=np.float32)

    # metadata
    meta = {
        "n_bars": int(n),
        "lookback": int(lookback),
        "horizon": int(horizon),
        "feature_names": feature_names,
        "label_mode": label_mode,
    }
    return X, y, meta


def write_outputs(out_dir: str, X: np.ndarray, y: np.ndarray, meta: dict, prices_df: pd.DataFrame):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    feats_path = p / "features.npy"
    labels_path = p / "labels.npy"
    meta_path = p / "features_meta.json"
    prices_path = p / "prices.csv"

    np.save(str(feats_path), X)
    np.save(str(labels_path), y)
    # ensure JSON serializable (convert numpy ints if present)
    meta_serializable = json.loads(json.dumps(meta, default=lambda o: int(o) if hasattr(o, "item") else str(o)))
    with open(str(meta_path), "w", encoding="utf-8") as f:
        json.dump(meta_serializable, f, indent=2)
    # save cleaned prices (ISO datetime)
    prices_df.to_csv(str(prices_path), index=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="cli generate-features", description="Generate features/labels from CSV price data")
    parser.add_argument("--csv", required=True, help="Input historical CSV path (must contain time,open,high,low,close,volume)")
    parser.add_argument("--out", required=True, help="Output directory (will contain features.npy, labels.npy, features_meta.json)")
    parser.add_argument("--lookback", type=int, default=10, help="Number of lookback bars for features (default 10)")
    parser.add_argument("--horizon", type=int, default=1, help="Label horizon in bars (default 1)")
    parser.add_argument("--label-mode", choices=("direction", "return"), default="direction", help="Label generation mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for deterministic processes")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    csv_path = resolve_path(args.csv)
    if csv_path is None:
        logger.error("CSV path not found: %s", args.csv)
        return 2

    logger.info("Loading CSV from %s", csv_path)
    prices_df = load_csv_prices(csv_path)

    # build features
    logger.info("Building features: lookback=%s horizon=%s", args.lookback, args.horizon)
    X, y, meta = build_features_and_labels(prices_df, lookback=args.lookback, horizon=args.horizon, label_mode=args.label_mode)

    # attach some provenance
    meta["source_csv"] = os.path.abspath(csv_path)
    meta["generator_version"] = "v1"
    meta["seed"] = int(args.seed)

    logger.info("Output: X.shape=%s y.shape=%s writing to %s", X.shape, y.shape, args.out)
    write_outputs(args.out, X, y, meta, prices_df)
    logger.info("Wrote features to %s (features.npy, labels.npy, features_meta.json)", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
