# ai/models/transformer.py
"""
Small, lightweight stub for TransformerModel used to satisfy imports during
CLI / offline training runs. This is intentionally minimal and test-friendly.

Replace with your real transformer implementation (PyTorch/Flax) when you
integrate the full model. The stub provides the same symbol and a tiny API:
- TransformerModel(config)
- fit(X, y)    # optional; no-op
- predict(X) -> np.ndarray
- save(path) / load(path)
"""

from __future__ import annotations
from typing import Any, Optional
import numpy as np
import json
import os

class TransformerModel:
    def __init__(self, config: Optional[dict] = None):
        # store config for debugging; keep model state simple
        self.config = dict(config or {})
        # small numpy "weights" placeholder for compatibility with save/load
        self._weights = None

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 32, verbose: bool = False):
        """
        Minimal no-op training routine that records shape information.
        This exists so code paths that call .fit() don't fail. It does not
        perform real learning—replace with a proper training loop later.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        # set a deterministic pseudo-weight to mark a "trained" state
        self._weights = {"x_mean": float(X.mean()) if X.size else 0.0, "y_mean": float(y.mean()) if y.size else 0.0}
        if verbose:
            print(f"[TransformerModel.stub] fit called: X.shape={X.shape} y.shape={y.shape} epochs={epochs}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return a deterministic dummy prediction (shape-compatible)."""
        X = np.asarray(X)
        n = X.shape[0] if X.ndim > 0 else 1
        # crude deterministic prediction using stored weights or zeros
        base = 0.0
        if self._weights:
            base = float(self._weights.get("y_mean", 0.0))
        # return zeros or base repeated (shape (n,))
        return np.full((n,), base, dtype=float)

    def save(self, path: str):
        """Serialize config+weights to JSON for now (quick compatibility)."""
        d = {"config": self.config, "weights": self._weights}
        ddir = os.path.dirname(path)
        if ddir and not os.path.exists(ddir):
            os.makedirs(ddir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f)

    @classmethod
    def load(cls, path: str) -> "TransformerModel":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        obj = cls(d.get("config", {}))
        obj._weights = d.get("weights")
        return obj
