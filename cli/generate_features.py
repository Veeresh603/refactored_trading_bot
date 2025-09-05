# cli/generate_features.py
"""
Small CLI to generate features.npy, labels.npy, features_meta.json from CSV.
Usage:
  python -m cli.generate_features --csv data/prices.csv --out data --lookback 10 --horizon 1
"""
import argparse
import os
import numpy as np
from data.loader import load_price_csv, build_features_labels, save_features_meta

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lookback", type=int, default=10)
    p.add_argument("--horizon", type=int, default=1)
    args = p.parse_args()

    df = load_price_csv(args.csv)
    X, y, meta = build_features_labels(df, lookback=args.lookback, label_horizon=args.horizon)
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "features.npy"), X)
    np.save(os.path.join(args.out, "labels.npy"), y)
    save_features_meta(os.path.join(args.out, "features_meta.json"), meta)
    print("Wrote features.npy, labels.npy and features_meta.json to", args.out)


if __name__ == "__main__":
    main()
