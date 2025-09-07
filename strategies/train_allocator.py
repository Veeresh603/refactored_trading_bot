"""
Train RL Allocator
------------------
- Trains PPO agent to act as allocator
- Observations: returns + Greeks
- Actions: choose strategy, strike idx, expiry idx, signal
- Reward: risk-adjusted PnL
- Uses Gymnasium API (reset -> obs,info and step -> obs,reward,terminated,truncated,info)
- Defensive: robust to execution_engine signature differences
- GPU-aware: will use CUDA device if PyTorch + CUDA available (minimal changes)
"""

import os
import inspect
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym  # gymnasium API (reset returns obs, info)
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

from core import execution_engine
from strategies.rl_allocator_callback import AllocatorMetricsCallback

logger = logging.getLogger("TrainAllocator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AllocatorEnv(gym.Env):
    """
    RL Environment for allocator.
    State: [Pnl normalized, delta, gamma, vega]
    Action: [strategy_idx, strike_idx, expiry_idx, signal]
    Reward: risk-adjusted PnL

    Gymnasium-compatible reset signature: reset(seed=None, options=None) -> obs, info
    step -> obs, reward, terminated, truncated, info
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        asset_strategies,
        strikes,
        expiries,
        spot_series,
        max_steps: int = 500,
        max_drawdown_penalty: float = 0.5,
        margin_penalty: float = 0.1,
        initial_balance: float = 100000.0,
    ):
        super().__init__()

        self.asset_strategies = list(asset_strategies)
        self.strikes = list(strikes)
        self.expiries = list(expiries)
        self.spot_series = np.asarray(spot_series, dtype=float)
        self.max_steps = min(int(max_steps), max(1, len(self.spot_series) - 1))

        # Spaces
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)
        # action: strategy_idx, strike_idx, expiry_idx, signal (0 flat,1 short,2 long)
        self.action_space = gym.spaces.MultiDiscrete([len(self.asset_strategies), len(self.strikes), len(self.expiries), 3])

        # State
        self.current_step = 0
        self.prev_equity = float(initial_balance)
        self.peak_equity = float(initial_balance)
        self.max_drawdown_penalty = float(max_drawdown_penalty)
        self.margin_penalty = float(margin_penalty)

        # init execution engine with initial balance if possible
        try:
            execution_engine.reset_engine(self.prev_equity)
        except Exception:
            logger.debug("execution_engine.reset_engine not available or failed", exc_info=True)

        # Randomness for environment, can be reseeded in reset
        self._rng = np.random.RandomState(0)

    # Gymnasium-compatible reset
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset environment and return (obs, info) as required by Gymnasium / SB3 env_checker.
        Accepts seed kwarg for determinism.
        """
        if seed is not None:
            try:
                self._rng = np.random.RandomState(int(seed))
            except Exception:
                self._rng = np.random.RandomState(0)

        self.current_step = 0
        # keep prev_equity if present, else init
        self.prev_equity = float(getattr(self, "prev_equity", 100000.0))
        self.peak_equity = float(self.prev_equity)
        try:
            execution_engine.reset_engine(self.prev_equity)
        except Exception:
            logger.debug("execution_engine.reset_engine not present or failed during reset", exc_info=True)

        # build initial observation (zeros or based on spot_series[0])
        obs = np.zeros(4, dtype=np.float32)
        info = {"equity": self.prev_equity}
        return obs, info

    def step(self, action):
        """
        Execute one step. action is a 4-tuple from MultiDiscrete:
        (strategy_idx, strike_idx, expiry_idx, signal_idx)
        signal_idx: 0 -> flat, 1 -> short, 2 -> long
        Returns: obs, reward, terminated, truncated, info
        """
        # increment step first (so step=1 corresponds to spot_series[1])
        self.current_step += 1
        if self.current_step >= len(self.spot_series):
            self.current_step = len(self.spot_series) - 1

        spot = float(self.spot_series[self.current_step])

        # Decode action
        try:
            strategy_idx, strike_idx, expiry_idx, signal_idx = list(map(int, action))
        except Exception:
            # invalid action shape; treat as no-op
            logger.debug("Invalid action received in step(): %s", action, exc_info=True)
            strategy_idx, strike_idx, expiry_idx, signal_idx = 0, 0, 0, 0

        # map signal_idx to trading signal value
        signal_val = 1 if signal_idx == 2 else -1 if signal_idx == 1 else 0

        strike_offset = self.strikes[int(strike_idx) % len(self.strikes)]
        expiry_type = self.expiries[int(expiry_idx) % len(self.expiries)]

        # Execution: place order only if not flat
        if signal_val != 0:
            strike = int(round(spot / 50.0) * 50 + float(strike_offset))
            symbol = f"NIFTY{strike}{'CE' if signal_val == 1 else 'PE'}"
            qty = 50
            sigma = 0.2
            is_call = signal_val == 1
            expiry_days = 30 if expiry_type == "monthly" else 7

            # Use a defensive wrapper that attempts to call execution_engine.place_order
            # in a way compatible with various signatures.
            self._safe_place_order(
                symbol=symbol,
                qty=qty,
                price=spot,
                strike=strike,
                sigma=sigma,
                is_call=is_call,
                expiry_days=expiry_days,
            )

        # Portfolio update & fetch status
        try:
            status = execution_engine.account_status(spot)
            equity = float(status.get("balance", 0.0) + status.get("total", 0.0))
            margin_used = float(status.get("margin_used", 0.0))
        except Exception:
            # execution engine missing or not configured; produce synthetic values
            equity = float(self.prev_equity)  # no change
            margin_used = 0.0

        # Reward: risk adjusted PnL
        pnl_change = equity - float(self.prev_equity)
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = self.peak_equity - equity

        reward = float(pnl_change)
        reward -= float(self.max_drawdown_penalty) * (drawdown / 1000.0)
        reward -= float(self.margin_penalty) * (margin_used / 1000.0)

        # Update state
        self.prev_equity = float(equity)

        # Try to get portfolio greeks; if unavailable, return zeros for greeks
        try:
            delta, gamma, vega, theta = execution_engine.portfolio_greeks(spot)
            delta = float(delta)
            gamma = float(gamma)
            vega = float(vega)
        except Exception:
            delta, gamma, vega = 0.0, 0.0, 0.0

        obs = np.array([pnl_change / 1000.0, delta, gamma, vega], dtype=np.float32)

        terminated = bool(self.current_step >= self.max_steps)
        truncated = False  # we don't implement truncation semantics separately here
        info = {"equity": equity, "drawdown": drawdown, "margin_used": margin_used}

        return obs, float(reward), terminated, truncated, info

    def render(self, mode="human"):
        print(f"Step {self.current_step} | PrevEquity={self.prev_equity:.2f} | Peak={self.peak_equity:.2f}")

    # Defensive wrapper for execution_engine.place_order
    def _safe_place_order(self, symbol, qty, price, strike, sigma, is_call, expiry_days):
        """
        Try to call execution_engine.place_order with compatible kwarg names.
        If execution_engine.place_order doesn't accept a 'strike' kw, attempt
        common alternatives or positional calls. All exceptions are caught and logged.
        """
        fn = getattr(execution_engine, "place_order", None)
        if fn is None:
            logger.debug("execution_engine.place_order not available; skipping order placement for %s", symbol)
            return None

        # Inspect signature
        try:
            sig = inspect.signature(fn)
            params = set(sig.parameters.keys())
        except Exception:
            params = set()

        base_kwargs = {
            "symbol": symbol,
            "qty": int(qty),
            "price": float(price),
            "sigma": float(sigma),
            "is_call": bool(is_call),
            "expiry_days": int(expiry_days),
        }

        # Common names for strike param across different engines
        strike_names = ["strike", "strike_price", "strike_px", "strikePrice", "strike_px"]

        # Try keyword variants
        for sname in strike_names:
            if sname in params:
                kwargs = dict(base_kwargs)
                kwargs[sname] = strike
                try:
                    return fn(**kwargs)
                except TypeError as e:
                    logger.debug("place_order(**%s) raised TypeError: %s", list(kwargs.keys()), e)
                    continue
                except Exception:
                    logger.exception("place_order(**%s) raised exception", list(kwargs.keys()))
                    return None

        # Try other accepted param names if present
        if "symbol" in params and "price" in params and "qty" in params:
            try:
                # If function accepts 'expiration' or similar, still try to call with base_kwargs
                kwargs = dict(base_kwargs)
                # only include known params present in signature
                kwargs = {k: v for k, v in kwargs.items() if k in params}
                # attach strike if any plausible param exists
                if "strike" in params:
                    kwargs["strike"] = strike
                return fn(**kwargs)
            except Exception:
                logger.debug("place_order(**selected_kwargs) raised; will try positional", exc_info=True)

        # Try positional calls (best-effort)
        try:
            # common positional layout: symbol, qty, price, strike, sigma, is_call, expiry_days
            return fn(symbol, int(qty), float(price), strike, float(sigma), bool(is_call), int(expiry_days))
        except TypeError:
            try:
                # try minimal signature: symbol, qty, price
                return fn(symbol, int(qty), float(price))
            except Exception:
                logger.exception("place_order positional fallbacks failed for symbol=%s", symbol)
                return None
        except Exception:
            logger.exception("place_order unexpected exception for symbol=%s", symbol)
            return None


def train_allocator(
    timesteps: int = 100000,
    save_path: str = "models/best_allocator_strike_expiry",
    log_dir: str = "logs/rl_allocator",
    data_path: str = "data/historical.csv",
    resume_from: Optional[str] = None,
    seed: int = 42,
):
    """
    Train PPO Allocator with risk-adjusted rewards + checkpointing.

    Args:
        timesteps (int): training steps
        save_path (str): final save path for PPO model
        log_dir (str): TensorBoard log dir
        data_path (str): historical data path
        resume_from (str): path to checkpoint (if resuming training)
        seed (int): RNG seed for reproducibility
    """
    import random

    # Try to import torch and detect CUDA device
    try:
        import torch

        torch_available = True
        cuda_available = torch.cuda.is_available()
    except Exception:
        torch = None
        torch_available = False
        cuda_available = False

    # seed python/numpy/torch if available
    random.seed(int(seed))
    np.random.seed(int(seed))
    if torch_available:
        try:
            torch.manual_seed(int(seed))
            if cuda_available:
                torch.cuda.manual_seed_all(int(seed))
                # optional: improve determinism if desired (may affect perf)
                try:
                    torch.backends.cudnn.deterministic = False
                    torch.backends.cudnn.benchmark = True
                except Exception:
                    pass
        except Exception:
            logger.debug("torch seeding failed", exc_info=True)

    # choose device for SB3 / PyTorch
    device = "cpu"
    if torch_available and cuda_available:
        device = "cuda"
    logger.info("PyTorch available=%s CUDA available=%s -> using device=%s", torch_available, cuda_available, device)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Load market data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path, parse_dates=["time"])
    if "close" not in df.columns:
        raise ValueError("data must contain 'close' column")
    spot_series = df["close"].values

    asset_strategies = ["SMA", "RSI", "MeanReversion"]
    strikes = [-100, 0, 100]
    expiries = ["weekly", "monthly"]

    env = AllocatorEnv(asset_strategies, strikes, expiries, spot_series, max_steps=500)

    # perform environment checks (will call reset(seed=...) and step with trials)
    try:
        check_env(env, warn=True)
    except Exception as e:
        # If check_env fails in some environments it's ok to continue but log
        logger.warning("check_env reported issues: %s. Continuing but consider fixing env API.", e)

    logger.info("🚀 Training RLAllocator for %s timesteps (risk-adjusted) on device=%s", timesteps, device)
    if resume_from and os.path.exists(resume_from):
        logger.info("🔄 Loading checkpoint from %s", resume_from)
        model = PPO.load(resume_from, env=env, verbose=1, device=device, _init_setup_model=True)
    else:
        # pass device explicitly so SB3 uses the requested device
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device=device)

    # Logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    # Attach callback that saves intermediate models
    callback = AllocatorMetricsCallback(save_path=os.path.join(save_path, "checkpoints"), verbose=1)

    # Train
    model.learn(total_timesteps=int(timesteps), callback=callback)

    # Save final model
    try:
        model.save(save_path)
        logger.info("✅ RLAllocator model saved at %s", save_path)
    except Exception:
        logger.exception("Failed to save PPO model to %s", save_path)

    logger.info("📊 TensorBoard logs available at %s", log_dir)
