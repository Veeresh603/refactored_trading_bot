# envs/trading_env.py
"""
TradingEnv with gym-like attributes for RL training.

- Tries to load data/historical.csv (columns: time,open,high,low,close,volume).
- Falls back to synthetic data if missing/invalid.
- Exposes:
    - observation_space.shape  (e.g. (5,))
    - action_space.n           (discrete actions: 0=hold,1=buy,2=sell)
    - reset() -> obs (np.ndarray)
    - step(action) -> (obs, reward, done, info)
    - get_price(), __len__()
- Compatible with scripts that use env.observation_space.shape[0] and env.action_space.n
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
from types import SimpleNamespace
from typing import Tuple

logger = logging.getLogger(__name__)
logging.getLogger("pandas").setLevel(logging.WARNING)


# Try to import gym spaces; if not available create a tiny shim
try:
    from gym import spaces
    GYM_AVAILABLE = True
except Exception:
    GYM_AVAILABLE = False

    class _DiscreteShim:
        def __init__(self, n):
            self.n = n

    class _BoxShim:
        def __init__(self, shape):
            self.shape = tuple(shape)


class TradingEnv:
    """
    Minimal trading environment suitable for RL training scripts.

    Observation: vector of [open, high, low, close, volume_norm]
    Actions (discrete n=3): 0 = hold, 1 = buy, 2 = sell
    """

    def __init__(
        self,
        data_path: str = "data/historical.csv",
        resample: str | None = None,
        synthetic_rows: int = 2000,
        feature_window: int = 1,
    ):
        """
        data_path : path to CSV (time,open,high,low,close,volume)
        resample  : optional pandas resample string (e.g. "1T")
        synthetic_rows : rows to generate if CSV missing
        feature_window : number of timesteps stacked into observation (default 1)
        """
        self.data_path = Path(data_path)
        self.resample = resample
        self.synthetic_rows = synthetic_rows
        self.feature_window = max(1, int(feature_window))

        # load data (or generate synthetic)
        self._load_data()

        # features per timestep: open, high, low, close, volume_norm
        self.num_features = 5
        self.obs_shape = (self.feature_window * self.num_features,)

        # action space
        self.num_actions = 3  # hold, buy, sell

        # create spaces (gym if available)
        if GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32
            )
            self.action_space = spaces.Discrete(self.num_actions)
        else:
            # simple fallback with required attributes
            self.observation_space = _BoxShim(self.obs_shape)
            self.action_space = _DiscreteShim(self.num_actions)

        # RL state
        self.current_step = 0
        self.done = False

        # simple bookkeeping
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0

    def _load_data(self):
        if self.data_path.exists():
            try:
                df = pd.read_csv(self.data_path, parse_dates=["time"])
                df = df.sort_values("time").set_index("time")
                required = {"open", "high", "low", "close", "volume"}
                if not required.issubset(df.columns):
                    raise ValueError(f"CSV missing required columns: {required - set(df.columns)}")
                if self.resample:
                    df = df.resample(self.resample).agg({
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }).dropna()
                self.data = df
                logger.info("Loaded historical data from %s (%d rows)", self.data_path, len(self.data))
                return
            except Exception as e:
                logger.warning("Failed to read %s: %s", self.data_path, e)

        # fallback synthetic
        logger.warning(
            "Data file %s not found or invalid — generating synthetic dataset (%d rows).",
            self.data_path, self.synthetic_rows
        )
        self.data = self._generate_synthetic(nb=self.synthetic_rows)

    def _generate_synthetic(self, nb=2000):
        start = pd.Timestamp("2025-01-01 09:15:00")
        idx = pd.date_range(start=start, periods=nb, freq="1min")
        rng = np.random.default_rng(seed=42)
        returns = rng.normal(loc=0.0, scale=0.0008, size=nb)
        price = 20000 * np.exp(np.cumsum(returns))

        opens = price + rng.normal(0, 0.5, size=nb)
        closes = price + rng.normal(0, 0.5, size=nb)
        highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 1.0, size=nb))
        lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 1.0, size=nb))
        vols = np.maximum(1, (rng.normal(loc=1000, scale=300, size=nb)).astype(int))

        df = pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols
        }, index=idx)
        df.index.name = "time"
        return df

    # ---- Helpers to build observation vector ----
    def _row_to_feature(self, row: pd.Series) -> np.ndarray:
        # normalize volume to log scale to reduce huge ranges
        vol_norm = np.log1p(float(row["volume"]))
        return np.asarray([row["open"], row["high"], row["low"], row["close"], vol_norm], dtype=np.float32)

    def _get_observation(self, step_idx: int) -> np.ndarray:
        # stack last `feature_window` rows (most recent last)
        start = max(0, step_idx - (self.feature_window - 1))
        rows = self.data.iloc[start: step_idx + 1]
        # If fewer rows than window, pad by repeating first row
        if len(rows) < self.feature_window:
            pad_count = self.feature_window - len(rows)
            pad_row = rows.iloc[0]
            pads = np.stack([self._row_to_feature(pad_row)] * pad_count, axis=0)
            body = np.stack([self._row_to_feature(r) for _, r in rows.iterrows()], axis=0)
            stacked = np.concatenate([pads, body], axis=0)
        else:
            stacked = np.stack([self._row_to_feature(r) for _, r in rows.iterrows()], axis=0)
        return stacked.flatten()

    # ---- Minimal gym-like API ----
    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.done = False
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        obs = self._get_observation(self.current_step)
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        action: int (0=hold, 1=buy, 2=sell)
        This env is a stub: reward is zero by default; your RL training scripts can replace reward logic.
        """
        # Advance time
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        # Placeholder reward logic: 0.0 (user should provide real reward rules)
        reward = 0.0

        # Update unrealized_pnl as naive mark-to-market (zero here; user can compute)
        self.unrealized_pnl = 0.0

        obs = self._get_observation(self.current_step)
        info = {
            "time": str(self.data.index[self.current_step]),
            "price": float(self.data["close"].iloc[self.current_step])
        }
        return obs, reward, self.done, info

    def get_price(self, idx: int | None = None) -> float:
        if idx is None:
            idx = self.current_step
        return float(self.data["close"].iloc[idx])

    def __len__(self):
        return len(self.data)

    # small convenience: allow attribute access that typical training scripts expect
    @property
    def shape(self):
        return self.obs_shape
