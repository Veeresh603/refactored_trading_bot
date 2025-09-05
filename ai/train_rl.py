# ai/train_rl.py
"""
Training entrypoint for the trading bot.

Features included in this patched file:
- CLI with modes: train, supervised-check
- --orderbook-csv and --tca-params CLI options to load an OrderBookSampler and TCA parameters
- SimpleSyntheticEnv with partial fills/orderbook integration
- RLTrainer that captures execution_model into run metadata and writes per-eval purged-CV JSON including features_meta
- Uses the lightweight RLAgent implemented elsewhere (ai/models/rl_agent.py)
- Synthetic dataset generation for supervised-check
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import pickle
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# local project imports
from backtesting.orderbook_loader import load_orderbook_csv
from backtesting.orderbook_sampler import OrderBookSampler
from backtesting.tca import estimate_impact_from_params
from backtesting.backtest import Backtester  # for supervised-check mode
from ai.models.rl_agent import RLAgent  # lightweight REINFORCE agent in repo
from core.validation.purged_cv import purged_kfold_indices  # purged CV utilities (existing)

logger = logging.getLogger("TradingBot.TrainRL")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


# ------------------------
# Config dataclass
# ------------------------
@dataclass
class TrainConfig:
    seed: int = 42
    episodes: int = 200
    steps_per_episode: int = 100
    lr: float = 1e-3
    gamma: float = 0.99
    checkpoint_dir: str = "checkpoints"
    val_lookahead: int = 5
    val_n_splits: int = 5
    val_embargo: float = 0.01
    supervised_features: Optional[str] = None
    supervised_labels: Optional[str] = None
    use_synthetic_val: bool = False
    synthetic_val_samples: int = 500
    synthetic_val_features: int = 10


# ------------------------
# Synthetic RL Environment (improved)
# ------------------------
from typing import Iterable
import math


class SimpleSyntheticEnv:
    """
    Synthetic trading environment supporting:
      - partial fills over multiple steps
      - orderbook sampler fallback
      - TCA-driven impact estimates
      - action mapping: 0=HOLD, 1=BUY, 2=SELL
    """

    def __init__(
        self,
        window: int = 5,
        episode_length: int = 200,
        price_drift: float = 0.0,
        seed: Optional[int] = None,
        fill_delay_steps: int = 1,
        liquidity_fraction: float = 0.1,
        slippage_pct: float = 0.0,
        commission: float = 0.0,
        orderbook_sampler: Optional[OrderBookSampler] = None,
        tca_params: Optional[Dict[str, float]] = None,
    ):
        self.window = int(window)
        self.episode_length = int(episode_length)
        self.price_drift = float(price_drift)
        self.current_step = 0
        self.prices: List[float] = []
        self.position = 0  # -1,0,1
        self.fill_delay_steps = int(fill_delay_steps)
        self.liquidity_fraction = float(liquidity_fraction)
        self.slippage_pct = float(slippage_pct)
        self.commission = float(commission)
        self.orderbook_sampler = orderbook_sampler
        self.tca_params = tca_params or {"a": 0.0, "b": 0.0}
        self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self._order_queue: List[Dict[str, Any]] = []
        self.volumes: Optional[List[float]] = None

        # small action space shim
        class ASpace:
            def __init__(self, n):
                self.n = n

        self.observation_space = np.zeros((self.window,), dtype=np.float32)
        self.action_space = ASpace(3)

    def reset(self) -> np.ndarray:
        self.current_step = 0
        base = 100.0
        self.prices = [base + float(self._rng.normal(scale=0.1)) for _ in range(self.window)]
        self.position = 0
        self._order_queue.clear()
        self.volumes = [100.0 + float(self._rng.randint(0, 50)) for _ in range(self.episode_length + self.window + 10)]
        return np.array(self.prices[-self.window:], dtype=np.float32)

    def _enqueue_order_for_action(self, action: int, requested_units: float = 1.0) -> bool:
        # 0 => HOLD, 1 => BUY target +1, 2 => SELL target -1
        if action == 0:
            return False
        if action == 1:
            target = 1
        elif action == 2:
            target = -1
        else:
            return False

        if target != self.position:
            # create an order for the delta to reach target at requested_units scale
            # requested_units are absolute units for position sizing
            target_units = requested_units if target == 1 else -requested_units
            delta_units = target_units - (self.position * requested_units if self.position != 0 else 0.0)
            order = {
                "enqueue_step": int(self.current_step),
                "target": int(target),
                "requested_abs": float(abs(delta_units)),
                "signed_remaining": float(delta_units),
            }
            self._order_queue.append(order)
            return True
        return False

    def _generate_next_price(self) -> float:
        last = self.prices[-1]
        noise = float(self._rng.normal(scale=0.5))
        new_price = last * (1.0 + self.price_drift + noise * 0.001)
        return float(new_price)

    def step(self, action: int, requested_units: float = 1.0) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Apply action at time t: optionally enqueue an order.
        Advance to t+1, process fills eligible this bar, return (obs, reward, done, info).
        """
        ordered = self._enqueue_order_for_action(action, requested_units=requested_units)

        # Advance time & price
        new_price = self._generate_next_price()
        self.prices.append(new_price)
        self.current_step += 1
        bar_idx = self.current_step

        # determine bar volume (if orderbook sampler present we'll use it)
        bar_vol = float(self.volumes[bar_idx]) if self.volumes is not None else 0.0

        remaining_queue: List[Dict[str, Any]] = []
        fill_occurred = False
        last_fill_info = None

        # process FIFO order queue
        for order in self._order_queue:
            if bar_idx < (order["enqueue_step"] + self.fill_delay_steps):
                remaining_queue.append(order)
                continue

            needed_abs = float(order["requested_abs"])
            if needed_abs <= 0.0:
                continue

            # If orderbook_sampler available, use it
            if self.orderbook_sampler is not None:
                side = int(order["target"])
                ob_res = self.orderbook_sampler.execute(bar_idx, side, needed_abs)
                exec_abs = float(ob_res["executed"])
                vwap = ob_res["vwap"]
            else:
                avail = max(0.0, bar_vol * self.liquidity_fraction)
                exec_abs = min(needed_abs, avail)
                # Use TCA params to scale impact if provided
                impact_frac = estimate_impact_from_params(exec_abs, bar_vol if bar_vol > 0 else 1.0, self.tca_params)
                base_price = float(self.prices[-1])
                if int(order["target"]) == 1:
                    vwap = base_price * (1.0 + self.slippage_pct + impact_frac)
                else:
                    vwap = base_price * (1.0 - self.slippage_pct - impact_frac)

            if exec_abs <= 0.0:
                # nothing executable this bar; retain order
                remaining_queue.append(order)
                continue

            # executed signed units
            sign = 1.0 if order["signed_remaining"] > 0 else -1.0
            executed_signed_units = exec_abs * sign

            # update position: for simplicity we set to sign if any executed; position is coarse -1/0/1
            if executed_signed_units > 0:
                self.position = 1
            elif executed_signed_units < 0:
                self.position = -1

            # bookkeeping: update remaining units
            order["requested_abs"] = max(0.0, order["requested_abs"] - exec_abs)
            order["signed_remaining"] = (order["requested_abs"] if sign > 0 else -order["requested_abs"])

            # penalize reward by slippage and commission upon fill
            fill_occurred = True
            slippage_amt = abs(vwap - float(self.prices[-1])) if vwap is not None else 0.0
            commission_charged = float(self.commission) * (exec_abs / (abs(exec_abs) + 1e-12))

            last_fill_info = {
                "enqueue_step": int(order["enqueue_step"]),
                "fill_step": int(bar_idx),
                "executed_units": float(exec_abs),
                "remaining_units": float(order["requested_abs"]),
                "vwap": float(vwap) if vwap is not None else None,
                "slippage": float(slippage_amt),
                "commission": float(commission_charged),
            }

            # if still remaining, keep in queue
            if order["requested_abs"] > 0.0:
                remaining_queue.append(order)
            # else order completed

        self._order_queue = remaining_queue

        # compute reward: simple unrealized PnL + cost on fills
        prev_price = float(self.prices[-2])
        price_change = float(self.prices[-1] - prev_price)
        reward = float(price_change * self.position)
        if fill_occurred and last_fill_info is not None:
            reward -= last_fill_info["slippage"]
            reward -= last_fill_info["commission"]

        obs = np.array(self.prices[-self.window :], dtype=np.float32)
        done = self.current_step >= self.episode_length
        info = {
            "ordered": bool(ordered),
            "fill": bool(fill_occurred),
            "fill_info": last_fill_info,
            "position": int(self.position),
            "step": int(self.current_step),
            "price": float(self.prices[-1]),
        }
        return obs, reward, done, info


# ------------------------
# RL Trainer
# ------------------------
class RLTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        self.best_eval_reward = -float("inf")
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        # run metadata - always present
        self.run_metadata: Dict[str, Any] = {}
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_metadata["run_id"] = self.run_id
        # config snapshot
        try:
            if hasattr(cfg, "__dict__"):
                self.run_metadata["cfg"] = dict(vars(cfg))
            else:
                self.run_metadata["cfg"] = repr(cfg)
        except Exception:
            self.run_metadata["cfg"] = repr(cfg)

        # attempt to pre-load features_meta near supervised features
        self.features_meta = None
        if cfg.supervised_features:
            sf = cfg.supervised_features
            if sf and os.path.exists(sf):
                feat_dir = os.path.dirname(sf)
                meta_candidates = [
                    os.path.join(feat_dir, "features_meta.json"),
                    os.path.join(feat_dir, "features_meta.yaml"),
                    os.path.join("data", "features_meta.json"),
                    os.path.join(self.cfg.checkpoint_dir, "features_meta.json"),
                ]
                for mc in meta_candidates:
                    if os.path.exists(mc):
                        try:
                            with open(mc, "r", encoding="utf-8") as mf:
                                self.features_meta = json.load(mf)
                            logger.info("Pre-loaded features_meta from %s", mc)
                            break
                        except Exception:
                            logger.exception("Failed to parse features_meta at %s; ignoring", mc)

        # ensure execution_model defaults exist
        self.run_metadata.setdefault("execution_model", {"fill_delay_steps": None, "slippage_pct": None, "commission": None})

    # make_env used in training; accepts optional orderbook_sampler and tca_params
    def _make_env(self, orderbook_sampler: Optional[OrderBookSampler] = None, tca_params: Optional[Dict[str, float]] = None) -> SimpleSyntheticEnv:
        env = SimpleSyntheticEnv(
            window=5,
            episode_length=self.cfg.steps_per_episode,
            price_drift=0.0,
            seed=self.cfg.seed,
            fill_delay_steps=1,
            liquidity_fraction=0.1,
            slippage_pct=0.0,
            commission=0.0,
            orderbook_sampler=orderbook_sampler,
            tca_params=tca_params,
        )
        # capture execution_model into run_metadata
        try:
            exec_model = {
                "fill_delay_steps": getattr(env, "fill_delay_steps", None),
                "slippage_pct": getattr(env, "slippage_pct", None),
                "commission": getattr(env, "commission", None),
            }
            self.run_metadata["execution_model"] = exec_model
            logger.info("Captured execution_model: %s", exec_model)
        except Exception:
            logger.exception("Failed to capture execution_model from env")
        return env

    def _make_agent(self, obs_space_shape, action_space_n) -> RLAgent:
        # our RLAgent expects obs_dim (int) and act_dim (int)
        return RLAgent(int(obs_space_shape[0]), int(action_space_n), lr=self.cfg.lr, gamma=self.cfg.gamma, seed=self.cfg.seed)

    def train(self, orderbook_sampler: Optional[OrderBookSampler] = None, tca_params: Optional[Dict[str, float]] = None):
        env = self._make_env(orderbook_sampler=orderbook_sampler, tca_params=tca_params)
        obs_dim = env.observation_space.shape[0] if hasattr(env.observation_space, "shape") else len(env.observation_space)
        act_n = env.action_space.n if hasattr(env.action_space, "n") else env.action_space

        agent = self._make_agent((obs_dim,), act_n)
        logger.info("✅ RLAgent instantiated")

        # training loop (very simple REINFORCE episodic training)
        for ep in range(1, self.cfg.episodes + 1):
            obs = env.reset()
            total_reward = 0.0
            for step in range(self.cfg.steps_per_episode):
                action = agent.select_action(obs, deterministic=False)
                next_obs, reward, done, info = env.step(action)
                agent.store_transition(obs, action, reward)
                obs = next_obs
                total_reward += reward
                if done:
                    break
            agent.update()
            logger.info("Episode %d reward=%.4f steps=%d", ep, total_reward, step + 1)

            # periodic evaluation / supervised check at certain epochs (e.g., every 50)
            if ep % 50 == 0 or ep == self.cfg.episodes:
                # run supervised purged-CV check if supervised features provided
                self._supervised_eval_and_checkpoint(agent, ep, orderbook_sampler, tca_params)

        # final write of run metadata
        self.run_metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        run_path = os.path.join(self.cfg.checkpoint_dir, f"run_metadata_{self.run_id}.json")
        with open(run_path, "w", encoding="utf-8") as fh:
            json.dump(self.run_metadata, fh, indent=2)
        logger.info("Saved run metadata to %s", run_path)

    def _supervised_eval_and_checkpoint(self, agent: RLAgent, ep: int, orderbook_sampler: Optional[OrderBookSampler], tca_params: Optional[Dict[str, float]]):
        # If supervised features/labels are available, load and run purged-CV; otherwise optionally generate synthetic
        features_path = self.cfg.supervised_features
        labels_path = self.cfg.supervised_labels
        X = None
        y = None
        features_meta_content = None
        if features_path and os.path.exists(features_path) and labels_path and os.path.exists(labels_path):
            try:
                X = np.load(features_path)
                y = np.load(labels_path)
                # try to load features_meta near features file
                feat_meta_path = os.path.join(os.path.dirname(features_path), "features_meta.json")
                if os.path.exists(feat_meta_path):
                    with open(feat_meta_path, "r", encoding="utf-8") as fm:
                        features_meta_content = json.load(fm)
            except Exception:
                logger.exception("Failed to load supervised features/labels from provided paths; skipping supervised eval")
                X, y = None, None

        if X is None and self.cfg.use_synthetic_val:
            logger.info("Features/labels files not provided or missing. Generating synthetic dataset.")
            n = self.cfg.synthetic_val_samples
            d = self.cfg.synthetic_val_features
            rng = np.random.RandomState(self.cfg.seed)
            X = rng.randn(n, d)
            y = rng.randint(0, 2, size=(n,))
            features_meta_content = {"synthetic": True, "n": n, "d": d, "seed": int(self.cfg.seed)}
            logger.info("Generated synthetic dataset: n=%d d=%d seed=%d", n, d, int(self.cfg.seed))

        if X is None:
            logger.warning("Supervised eval skipped: no features available")
            return

        # Purged CV evaluation (log fold scores)
        fold_meta = []
        for fold_idx, (train_idx, test_idx) in enumerate(purged_kfold_indices(len(X), n_splits=self.cfg.val_n_splits, lookahead=self.cfg.val_lookahead, embargo=self.cfg.val_embargo)):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            clf = LogisticRegression(solver="liblinear", max_iter=200)
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[test_idx])
            score = float(accuracy_score(y[test_idx], preds))
            fold_meta.append({"fold": fold_idx + 1, "score": score, "train_size": int(len(train_idx)), "test_size": int(len(test_idx))})
            logger.info("Supervised eval fold %d score=%.4f train_size=%d test_size=%d", fold_idx + 1, score, len(train_idx), len(test_idx))

        # write per-ep purged CV metadata + embed features_meta + execution_model
        payload = {
            "ep": int(ep),
            "run_id": self.run_id,
            "folds": fold_meta,
            "features_meta": features_meta_content,
            "execution_model": self.run_metadata.get("execution_model", None),
            "val_params": {"lookahead": self.cfg.val_lookahead, "n_splits": self.cfg.val_n_splits, "embargo": self.cfg.val_embargo},
        }
        out_path = os.path.join(self.cfg.checkpoint_dir, f"purged_cv_folds_train_eval_ep{ep}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Wrote supervised purged-CV metadata to %s", out_path)


# ------------------------
# CLI entrypoint
# ------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train RL agent or run supervised checks")
    p.add_argument("--mode", type=str, default="train", choices=("train", "supervised-check"), help="Mode")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--supervised-features", type=str, default=None)
    p.add_argument("--supervised-labels", type=str, default=None)
    p.add_argument("--use-synthetic-val", action="store_true")
    p.add_argument("--synthetic-samples", type=int, default=500)
    p.add_argument("--synthetic-features", type=int, default=10)
    p.add_argument("--orderbook-csv", type=str, default=None, help="CSV with per-bar orderbook snapshots (book column)")
    p.add_argument("--tca-params", type=str, default=None, help="JSON file with tca params {a,b}")
    return p


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = TrainConfig(
        seed=args.seed,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        checkpoint_dir=args.checkpoint_dir,
        supervised_features=args.supervised_features,
        supervised_labels=args.supervised_labels,
        use_synthetic_val=args.use_synthetic_val,
        synthetic_val_samples=args.synthetic_samples,
        synthetic_val_features=args.synthetic_features,
    )

    # load orderbook and tca params if provided
    orderbook_sampler = None
    if args.orderbook_csv:
        try:
            levels = load_orderbook_csv(args.orderbook_csv, book_col="book")
            orderbook_sampler = OrderBookSampler(levels)
            logger.info("OrderBookSampler created from %s (bars=%d)", args.orderbook_csv, len(levels))
        except Exception:
            logger.exception("Failed to create OrderBookSampler; falling back to liquidity_fraction model")
            orderbook_sampler = None

    tca_params = None
    if args.tca_params:
        try:
            with open(args.tca_params, "r", encoding="utf-8") as fh:
                tca_params = json.load(fh)
            logger.info("Loaded tca_params from %s: %s", args.tca_params, tca_params)
        except Exception:
            logger.exception("Failed to load tca_params; ignoring")
            tca_params = None

    trainer = RLTrainer(cfg)
    if args.mode == "train":
        trainer.train(orderbook_sampler=orderbook_sampler, tca_params=tca_params)
    elif args.mode == "supervised-check":
        # quick supervised check flow using the trainer's eval helper
        trainer._supervised_eval_and_checkpoint(agent=None, ep=0, orderbook_sampler=orderbook_sampler, tca_params=tca_params)
        logger.info("Supervised purged-CV check complete")
    else:
        raise SystemExit("Unknown mode")


if __name__ == "__main__":
    main()
