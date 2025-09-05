# tests/test_loader.py
import os
import numpy as np
import pandas as pd
import tempfile
from data.loader import load_price_csv, build_features_labels, save_features_meta

def make_sample_csv(path):
    df = pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=30, freq="min").astype(str),
        "open": np.linspace(100, 130, 30),
        "high": np.linspace(101, 131, 30),
        "low": np.linspace(99, 129, 30),
        "close": np.linspace(100, 130, 30),
        "volume": np.linspace(10, 100, 30),
    })
    df.to_csv(path, index=False)

def test_load_and_build_features(tmp_path):
    p = tmp_path / "prices.csv"
    make_sample_csv(str(p))
    df = load_price_csv(str(p))
    assert "close" in df.columns
    X, y, meta = build_features_labels(df, lookback=5, label_horizon=1)
    assert X.ndim == 2
    assert len(y) == X.shape[0]
    assert "n_features" in meta
    outdir = tmp_path / "out"
    outdir.mkdir()
    save_features_meta(str(outdir / "features_meta.json"), meta)
    assert (outdir / "features_meta.json").exists()
