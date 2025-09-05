#!/usr/bin/env python3
"""
Build features.npy and labels.npy from an OHLCV CSV.

Usage:
  python scripts/make_features_labels.py --csv data/historical.csv --out-dir data --lookahead 5 --roll-windows 5 10 20 --rthreshold 0.0 --save-meta

Outputs:
  - data/features.npy
  - data/labels.npy
  - data/features_meta.json
  - data/leak_report.json
"""
import os
import json
import argparse
import numpy as np
import pandas as pd

# optional imports for advanced TA; code falls back to pure-pandas if missing
try:
    import pandas_ta as ta  # type: ignore
    HAVE_PANDAS_TA = True
except Exception:
    HAVE_PANDAS_TA = False

from datetime import datetime


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.rolling(window=window, min_periods=window).mean()
    ma_down = down.rolling(window=window, min_periods=window).mean()
    rs = ma_up / (ma_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line.fillna(0.0), signal_line.fillna(0.0), hist.fillna(0.0)


def leakage_report(X_df: pd.DataFrame, future_ret: pd.Series, top_k: int = 10):
    """
    Simple leakage detection: compute Pearson corr and Mutual Information against future_ret.
    Returns a dict with top_k suspicious features by MI and by abs(correlation).
    """
    out = {"by_corr": [], "by_mi": []}
    # Pearson correlation
    corr = X_df.apply(lambda col: float(col.corr(future_ret)))
    corr_abs = corr.abs().sort_values(ascending=False)
    out["by_corr"] = [{"feature": f, "corr": float(corr.loc[f])} for f in corr_abs.index[:top_k]]

    # Mutual information (discretize continuous future_ret into 2 bins: up/down)
    # We'll compute MI between X columns and the sign label
    try:
        from sklearn.feature_selection import mutual_info_classif
        y_bin = (future_ret > 0).astype(int).values
        mi_vals = mutual_info_classif(X_df.values, y_bin, discrete_features=False)
        mi_series = pd.Series(mi_vals, index=X_df.columns).sort_values(ascending=False)
        out["by_mi"] = [{"feature": f, "mi": float(mi_series.loc[f])} for f in mi_series.index[:top_k]]
    except Exception as e:
        out["by_mi"] = {"error": f"mutual_info_classif error: {e}"}
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="Input OHLCV CSV path (must include time and close)")
    p.add_argument("--out-dir", type=str, default="data", help="Where to write features.npy/labels.npy/meta")
    p.add_argument("--lookahead", type=int, default=5, help="Forward horizon (rows) for label")
    p.add_argument("--rthreshold", type=float, default=0.0, help="Threshold on future return to set label=1")
    p.add_argument("--roll-windows", nargs="+", type=int, default=[5, 10, 20], help="Rolling windows for MAs/std")
    p.add_argument("--save-meta", action="store_true", help="Save JSON metadata next to outputs")
    p.add_argument("--topk-leak", type=int, default=10, help="Top-k features for leak report")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv, parse_dates=[0], infer_datetime_format=True)
    # detect time & close columns
    time_col_candidates = ["time", "timestamp", df.columns[0]]
    time_col = None
    for c in time_col_candidates:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise SystemExit("No time/timestamp column found in CSV")
    if "close" not in df.columns:
        alt = None
        for c in ("Close", "close_price", "adj_close", "price"):
            if c in df.columns:
                alt = c; break
        if alt:
            df = df.rename(columns={alt: "close"})
        else:
            raise SystemExit("CSV must contain a 'close' column")

    df = df.sort_values(time_col).reset_index(drop=True)

    # base features
    for w in args.roll_windows:
        df[f"ma_{w}"] = df["close"].rolling(window=w, min_periods=1).mean()
        df[f"std_{w}"] = df["close"].rolling(window=w, min_periods=1).std().fillna(0.0)

    df["ret_1"] = df["close"].pct_change().fillna(0.0)
    for w in args.roll_windows:
        df[f"ret_{w}"] = df["close"].pct_change(periods=w).fillna(0.0)
    df["mom_5_20"] = df["ma_5"] - df["ma_20"]

    # RSI/MACD
    if HAVE_PANDAS_TA:
        try:
            df["rsi_14"] = ta.rsi(df["close"], length=14).fillna(50.0)
            macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
            df["macd_line"] = macd.iloc[:, 0].fillna(0.0)
            df["macd_signal"] = macd.iloc[:, 1].fillna(0.0)
            df["macd_hist"] = macd.iloc[:, 2].fillna(0.0)
        except Exception:
            df["rsi_14"] = compute_rsi(df["close"], window=14)  # fallback
            df["macd_line"], df["macd_signal"], df["macd_hist"] = compute_macd(df["close"])
    else:
        df["rsi_14"] = compute_rsi(df["close"], window=14)
        df["macd_line"], df["macd_signal"], df["macd_hist"] = compute_macd(df["close"])

    # labels: future return
    df["future_close"] = df["close"].shift(-args.lookahead)
    df["future_ret"] = (df["future_close"] - df["close"]) / df["close"]
    df["label"] = (df["future_ret"] > args.rthreshold).astype(int)

    df_valid = df.iloc[:-args.lookahead].copy()
    exclude = [time_col, "open", "high", "low", "close", "volume", "future_close", "future_ret", "label"]
    feature_cols = [c for c in df_valid.columns if c not in exclude]
    X = df_valid[feature_cols].values.astype(np.float32)
    y = df_valid["label"].values.astype(np.int8)

    # validation
    if X.ndim != 2:
        raise SystemExit(f"Expected feature matrix nxd, got shape {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise SystemExit(f"Mismatch samples X={X.shape[0]} vs y={y.shape[0]}")

    np.save(os.path.join(args.out_dir, "features.npy"), X)
    np.save(os.path.join(args.out_dir, "labels.npy"), y)
    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_cols": feature_cols,
        "lookahead": args.lookahead,
        "rthreshold": args.rthreshold,
        "roll_windows": args.roll_windows,
        # git sha: placeholder; CI should fill this if desired
        "git_sha": os.environ.get("GIT_SHA", None)
    }
    if args.save_meta:
        with open(os.path.join(args.out_dir, "features_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
    # leakage report
    leak = leakage_report(pd.DataFrame(X, columns=feature_cols), df_valid["future_ret"], top_k=args.topk_leak)
    with open(os.path.join(args.out_dir, "leak_report.json"), "w") as fh:
        json.dump(leak, fh, indent=2)

    print("Saved features.npy, labels.npy, features_meta.json (if requested), leak_report.json")
    print("X.shape=", X.shape, "y.shape=", y.shape)


if __name__ == "__main__":
    main()
