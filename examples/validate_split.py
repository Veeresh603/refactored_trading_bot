# examples/validate_split.py
"""
Example script to inspect purged k-fold indices for your dataset.

CSV expected columns: timestamp (optional), label, plus optional feature columns.
Usage:
  python examples/validate_split.py data/mydata.csv --label-col label --lookahead 5 --n-splits 5
"""
import argparse
import pandas as pd
import numpy as np
from core.validation.purged_cv import purged_kfold_indices

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=str)
    p.add_argument("--label-col", type=str, default="label")
    p.add_argument("--lookahead", type=int, default=5)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--embargo", type=float, default=0.01)
    return p.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    if args.label_col not in df.columns:
        raise SystemExit(f"Label column {args.label_col} not found in CSV")
    n = len(df)
    print(f"Loaded {n} rows from {args.csv}")
    for i, (train_idx, test_idx) in enumerate(purged_kfold_indices(n, n_splits=args.n_splits, lookahead=args.lookahead, embargo=args.embargo)):
        print(f"Fold {i+1}: train={len(train_idx)} test={len(test_idx)}")
        # Show the first few test indices and their labels
        labels = df[args.label_col].values
        print("  test sample labels (first 8):", labels[test_idx[:8]])
        # show how many train rows were purged for this fold:
        # naive train would be n - len(test_idx) (without embargo/purge)
        naive_train_size = n - len(test_idx)
        purged_count = naive_train_size - len(train_idx)
        print(f"  purged_count={purged_count}")
    print("Done")

if __name__ == '__main__':
    main()
