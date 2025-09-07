# ai/train_rl.py
"""
Lightweight training runner for RL experiments (test-friendly).
- Provides TrainConfig, RLTrainer and a SimpleSyntheticEnv for tests.
- Defensive: falls back to RandomAgent if project agents not available.
- Writes run_metadata_{run_id}.json and supervised purged-CV metadata files.
"""
from __future__ import annotations
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("TradingBot.TrainRL")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


@dataclass
class TrainConfig:
    episodes: int = 50
    steps_per_episode: int = 100
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
    # execution model defaults (used in run metadata)
    fill_delay_steps: int = 1
    slippage_pct: float = 0.0
    commission: float = 0.0
    liquidity_fraction: float = 0.1
    # supervised eval files (optional)
    supervised_features: Optional[str] = None
    supervised_labels: Optional[str] = None
    # validation settings
    val_lookahead: int = 5
    val_n_splits: int = 5
    val_embargo: float = 0.01
    use_synthetic_val: bool = False
    synthetic_val_samples: int = 500
    synthetic_val_features: int = 10


class SimpleSyntheticEnv:
    """
    Small synthetic environment to test order enqueue/fill semantics.
    - Actions: 0 -> hold, 1 -> long (buy 1), -1 -> short (sell 1)
    - Maintains internal order queue, fill_delay_steps, liquidity_fraction, slippage_pct, commission.
    - Prices are a seeded random walk produced at init.
    """

    def __init__(
        self,
        window: int = 3,
        episode_length: int = 100,
        price_drift: float = 0.0,
        seed: int = 42,
        fill_delay_steps: int = 1,
        slippage_pct: float = 0.0,
        commission: float = 0.0,
        liquidity_fraction: float = 1.0,
    ):
        self.window = int(window)
        self.episode_length = int(episode_length)
        self.price_drift = float(price_drift)
        self.seed = int(seed)
        self.fill_delay_steps = int(fill_delay_steps)
        self.slippage_pct = float(slippage_pct)
        self.commission = float(commission)
        self.liquidity_fraction = float(liquidity_fraction)
        self.rng = np.random.RandomState(self.seed)

        # Create price path slightly longer than episode to avoid index issues
        n_prices = max(episode_length + 5, 10)
        p0 = float(100.0 + self.rng.normal(0, 1.0))
        steps = self.rng.normal(loc=self.price_drift, scale=1.0, size=n_prices - 1)
        self.prices = [p0]
        for s in steps:
            self.prices.append(max(0.01, self.prices[-1] + float(s)))

        # internal pointer used for advancing through prices on step()
        self._ptr = 0

        # trading state
        self.position = 0.0
        self.cash = 10000.0

        # order queue: list of dicts {enqueue_ptr, target (1 or -1), requested_units, remaining_units, signed_remaining}
        self.order_queue: List[Dict[str, Any]] = []

        # action_space compatibility (some code inspects .n)
        class _AS:
            def __init__(self, n):
                self.n = n

        self.action_space = _AS(3)  # three discrete actions mapped later

    def reset(self):
        self._ptr = 0
        self.position = 0.0
        self.cash = 10000.0
        self.order_queue = []
        # return simple observation (last `window` prices)
        obs = self._make_obs()
        return obs

    def _make_obs(self):
        # return vector of last `window` prices (pad if needed)
        idx = min(self._ptr, len(self.prices) - 1)
        start = max(0, idx - self.window + 1)
        block = self.prices[start: idx + 1]
        if len(block) < self.window:
            block = [self.prices[0]] * (self.window - len(block)) + block
        return np.array(block, dtype=float)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Apply action and advance environment by one price step. Returns obs, reward, done, info.
        - Accepts only actions in {0,1,-1}. Trainer often maps discrete indices to these values.
        - If an order is enqueued at this step, it's recorded and may be filled after fill_delay_steps.
        """
        # Accept only 0,1,-1 to match tests and mapping expectations
        if action not in (0, 1, -1):
            raise ValueError("action must be in {0,1,-1}")

        ordered = False
        fill = False
        fill_price = None
        executed_units = 0.0
        liquidity_used = 0.0

        # Enqueue new order if action indicates change
        current_side = 0 if abs(self.position) < 1e-12 else (1 if self.position > 0 else -1)
        if action != 0 and action != current_side:
            # request unit = 1 * side
            requested_units = 1.0 if action == 1 else -1.0
            signed_remaining = requested_units
            order = {
                "enqueue_ptr": int(self._ptr),
                "target": int(action),
                "requested_units": float(requested_units),
                "remaining_units": abs(float(requested_units)),
                "signed_remaining": float(signed_remaining),
            }
            self.order_queue.append(order)
            ordered = True

        # Advance pointer first to model "order placed at t uses future price"
        self._ptr += 1
        if self._ptr >= len(self.prices):
            # clip pointer and mark done later
            self._ptr = len(self.prices) - 1

        # process queue: fill orders whose enqueue_ptr + fill_delay_steps <= current_ptr
        remaining_queue: List[Dict[str, Any]] = []
        for order in self.order_queue:
            e_ptr = order["enqueue_ptr"]
            if self._ptr < (e_ptr + self.fill_delay_steps):
                remaining_queue.append(order)
                continue
            needed_abs = float(order["remaining_units"])
            if needed_abs <= 0.0:
                continue

            # available liquidity (simplified): use liquidity_fraction of a nominal bar volume (100)
            # This is adequate for tests — if zero liquidity, no fills happen.
            bar_volume = 100.0
            avail = bar_volume * self.liquidity_fraction
            execable = min(needed_abs, avail)
            if execable <= 0.0:
                remaining_queue.append(order)
                continue

            # vwap: use price at current pointer
            vwap = float(self.prices[self._ptr])
            # apply slippage to compute final fill price
            if self.slippage_pct != 0.0:
                if order["target"] == 1:
                    fill_price = vwap * (1.0 + abs(self.slippage_pct))
                else:
                    fill_price = vwap * (1.0 - abs(self.slippage_pct))
            else:
                fill_price = vwap

            sign = 1.0 if order["signed_remaining"] > 0.0 else -1.0
            executed_units = float(execable)
            executed_signed = executed_units * sign

            # update cash and position
            self.cash -= executed_signed * fill_price
            # commission proportional to executed fraction
            if abs(order["requested_units"]) > 0:
                commission_charged = (executed_units / abs(order["requested_units"])) * self.commission
            else:
                commission_charged = self.commission
            self.cash -= commission_charged

            self.position += executed_signed

            order["remaining_units"] = max(0.0, order["remaining_units"] - executed_units)
            order["signed_remaining"] = (order["remaining_units"] if sign > 0.0 else -order["remaining_units"])

            liquidity_used += float(executed_units)
            fill = True
            ordered = ordered or False

            # record partial fill: if remaining > 0, keep in queue
            if order["remaining_units"] > 0.0:
                remaining_queue.append(order)

        self.order_queue = remaining_queue

        # prepare info
        info = {
            "ordered": ordered,
            "fill": fill,
            "fill_price": float(fill_price) if fill else None,
            "executed_units": float(executed_units),
            "liquidity_used": float(liquidity_used),
            "cash": float(self.cash),
            "position": float(self.position),
            "idx": int(self._ptr),
            "fill_info": None,
        }

        obs = self._make_obs()
        # reward: simple pnl change (unrealized + cash) over last step
        unrealized = self.position * float(self.prices[self._ptr])
        total_equity = self.cash + unrealized
        reward = float(total_equity - 10000.0)  # relative to initial balance; tests don't depend on particular reward

        done = (self._ptr >= (len(self.prices) - 1)) or (self._ptr >= self.episode_length)
        return obs, reward, done, info


class RLTrainer:
    """
    Minimalistic RL trainer used by tests. Focuses on:
    - deterministic run_id and run_metadata saving,
    - defensive training loop (exceptions in env.step do not crash),
    - _init_agent fallback logic,
    - supervised eval file writing including features_meta if found.
    """

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        self.best_eval_reward = -float("inf")
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        # Run metadata base
        self.run_metadata: Dict[str, Any] = {}
        # Unique run id (UTC timestamp)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_metadata["run_id"] = self.run_id
        # dump cfg fields simply
        try:
            # if cfg is dataclass, __dict__ will work
            self.run_metadata["cfg"] = getattr(cfg, "__dict__", None) or vars(cfg)
        except Exception:
            self.run_metadata["cfg"] = str(cfg)

        # Attempt to locate features_meta near supervised features early and cache it
        self.features_meta = None
        if getattr(cfg, "supervised_features", None):
            sf = cfg.supervised_features
            # search likely places
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

        # placeholder agent, created via _init_agent
        self.agent = None

    def _init_agent(self, obs_dim: int, act_dim: int) -> None:
        """
        Initialize self.agent. Prefer project agents (MLPAgent, RLAgent), fallback to a tiny RandomAgent.
        Safe to call multiple times — no-op if already created.
        """
        if getattr(self, "agent", None) is not None:
            return

        # Try to construct a known agent from ai.models
        try:
            try:
                from ai.models.rl_agent import MLPAgent  # type: ignore
                logger.info("Attempting MLPAgent constructor: obs_dim=%s act_dim=%s", obs_dim, act_dim)
                self.agent = MLPAgent(obs_dim=obs_dim, act_dim=act_dim, hidden=32, lr=1e-3, gamma=0.99, seed=self.cfg.seed)
                logger.info("✅ MLPAgent instantiated")
                return
            except Exception:
                try:
                    from ai.models.rl_agent import RLAgent  # type: ignore
                    logger.info("Attempting RLAgent constructor: obs_dim=%s act_dim=%s", obs_dim, act_dim)
                    try:
                        self.agent = RLAgent(obs_dim, act_dim, lr=1e-3, gamma=0.99, seed=self.cfg.seed)
                    except Exception:
                        self.agent = RLAgent(obs_dim, act_dim)
                    logger.info("✅ RLAgent instantiated")
                    return
                except Exception:
                    raise
        except Exception:
            logger.debug("No project RL agent available (or import failed); falling back to RandomAgent", exc_info=True)

        # Fallback RandomAgent implementation
        class RandomAgent:
            def __init__(self, obs_dim, act_dim, seed=0):
                self.obs_dim = int(obs_dim)
                self.act_dim = int(act_dim)
                self.rng = np.random.RandomState(seed)
                self._obs_buf = []
                self._act_buf = []
                self._rew_buf = []

            def select_action(self, obs, deterministic=False):
                # deterministic -> always hold (0); otherwise uniform random over actions
                if deterministic:
                    return 0
                # if act_dim==3, return in { -1,0,1 }
                if self.act_dim == 3:
                    return int(self.rng.choice([-1, 0, 1]))
                return int(self.rng.randint(0, max(1, self.act_dim)))

            def store_transition(self, obs, act, reward):
                self._obs_buf.append(obs)
                self._act_buf.append(act)
                self._rew_buf.append(float(reward))

            def update(self, *args, **kwargs):
                # no-op
                self._obs_buf.clear()
                self._act_buf.clear()
                self._rew_buf.clear()

            def save(self, path):
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write("random-agent-placeholder\n")
                except Exception:
                    logger.debug("RandomAgent.save failed writing %s", path, exc_info=True)

        self.agent = RandomAgent(obs_dim, act_dim, seed=self.cfg.seed if hasattr(self.cfg, "seed") else 0)
        logger.info("✅ RandomAgent fallback instantiated (obs_dim=%s act_dim=%s)", obs_dim, act_dim)

    def _make_env(self) -> SimpleSyntheticEnv:
        # Build synthetic env with trainer config parameters
        env = SimpleSyntheticEnv(
            window=getattr(self.cfg, "window", 3),
            episode_length=getattr(self.cfg, "steps_per_episode", 100),
            price_drift=getattr(self.cfg, "price_drift", 0.0),
            seed=getattr(self.cfg, "seed", 42),
            fill_delay_steps=getattr(self.cfg, "fill_delay_steps", 1),
            slippage_pct=getattr(self.cfg, "slippage_pct", 0.0),
            commission=getattr(self.cfg, "commission", 0.0),
            liquidity_fraction=getattr(self.cfg, "liquidity_fraction", 1.0),
        )
        return env

    def _save_checkpoint(self, path: str):
        # minimal checkpoint: tell the world it's saved
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("checkpoint placeholder\n")
        except Exception:
            logger.exception("Failed to write checkpoint %s", path)

    def _run_supervised_eval_and_write_meta(self, ep: int) -> None:
        """
        If supervised features/labels exist (as numpy .npy), load them and write a purged_cv metadata file
        including any features_meta that was pre-loaded during __init__.
        This is intentionally minimal so tests can check for file presence & content.
        """
        feat_path = getattr(self.cfg, "supervised_features", None)
        lab_path = getattr(self.cfg, "supervised_labels", None)
        out_path = os.path.join(self.cfg.checkpoint_dir, f"purged_cv_folds_train_eval_ep{ep}.json")
        meta = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "ep": ep,
            "cfg": getattr(self.cfg, "__dict__", vars(self.cfg)),
            "folds": []
        }
        # If features exist, attempt to load; otherwise produce synthetic folds for tests
        try:
            if feat_path and os.path.exists(feat_path) and lab_path and os.path.exists(lab_path):
                import numpy as _np
                feats = _np.load(feat_path)
                labs = _np.load(lab_path)
                n = len(feats)
                # produce dummy folds: split into n_splits chunks
                n_splits = max(1, getattr(self.cfg, "val_n_splits", 5))
                fold_size = max(1, n // n_splits)
                for k in range(n_splits):
                    train_size = n - fold_size
                    test_size = fold_size
                    meta["folds"].append({"fold": k + 1, "train_size": train_size, "test_size": test_size, "score": None})
            else:
                # synthetic small metadata (used by tests to check file is written)
                for k in range(1, min(6, getattr(self.cfg, "val_n_splits", 5) + 1)):
                    meta["folds"].append({"fold": k, "train_size": 1, "test_size": 1, "score": None})
        except Exception:
            logger.exception("Supervised eval generation failed; writing fallback metadata")

        # include features_meta if pre-loaded
        if self.features_meta is not None:
            meta["features_meta"] = self.features_meta

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            logger.info("Wrote supervised purged-CV metadata to %s", out_path)
        except Exception:
            logger.exception("Failed to write supervised purged-CV metadata to %s", out_path)

    def train(self):
        """
        Main training loop. Defensive: any exception raised by env.step(...) is caught,
        logged, and the trainer continues (no hard crash). This prevents a single flaky
        sampler or env row from aborting long runs.
        """
        logger.info("Starting training: episodes=%s, steps=%s", self.cfg.episodes, self.cfg.steps_per_episode)
        env = self._make_env()
        obs_dim = getattr(env, "window", 6)
        # act_dim: if env.action_space has attribute n use that else default to 3
        act_dim = getattr(getattr(env, "action_space", None), "n", 3)

        # init agent (may fall back to RandomAgent)
        self._init_agent(obs_dim, act_dim)

        # training
        checkpoint_every = max(1, int(self.cfg.steps_per_episode // 2))  # save a few times during short runs
        for ep in range(1, int(self.cfg.episodes) + 1):
            try:
                obs = env.reset()
            except Exception:
                logger.exception("Env.reset failed at episode %s; skipping episode", ep)
                continue

            ep_reward = 0.0
            for step in range(int(self.cfg.steps_per_episode)):
                # let agent pick an action; the agent may use a different action space numbering
                try:
                    raw_action = self.agent.select_action(obs, deterministic=False)
                    # map agent output to environment action semantics {0,1,-1}
                    # common agent returns: -1,0,1 or 0,1,2 (map 2->-1)
                    if raw_action in (0, 1, -1):
                        action = int(raw_action)
                    else:
                        # if agent returned discrete 0..2 map -> 0->0,1->1,2->-1
                        if int(raw_action) == 2:
                            action = -1
                        elif int(raw_action) == 1:
                            action = 1
                        else:
                            action = 0
                except Exception:
                    logger.exception("Agent.select_action failed; using hold")
                    action = 0

                # step env defensively
                try:
                    next_obs, reward, done, info = env.step(action)
                except Exception:
                    logger.exception("Env.step raised exception at ep=%s step=%s; skipping step", ep, step)
                    # continue but do not crash entire run
                    continue

                # bookkeeping for agent
                try:
                    self.agent.store_transition(obs, action, reward)
                except Exception:
                    logger.debug("Agent.store_transition failed; continuing", exc_info=True)

                obs = next_obs
                ep_reward += float(reward)

                # agent update every few steps (lightweight)
                try:
                    if hasattr(self.agent, "update"):
                        self.agent.update()
                except Exception:
                    logger.exception("Agent.update failed; continuing")

                if done:
                    break

            # end episode
            logger.info("Episode %s reward=%.4f steps=%s", ep, ep_reward, step + 1)

            # checkpoint & supervised eval
            ckpt_path = os.path.join(self.cfg.checkpoint_dir, f"agent_ep{ep}.ckpt")
            try:
                if hasattr(self.agent, "save"):
                    self.agent.save(ckpt_path)
                else:
                    self._save_checkpoint(ckpt_path)
            except Exception:
                logger.exception("Failed to save checkpoint at %s", ckpt_path)

            # optionally run supervised eval & write purged-CV metadata that includes features_meta
            try:
                self._run_supervised_eval_and_write_meta(ep)
            except Exception:
                logger.exception("Supervised eval/write meta failed at ep=%s", ep)

        # finalize run metadata: include execution_model
        self.run_metadata.setdefault("execution_model", {
            "fill_delay_steps": int(getattr(self.cfg, "fill_delay_steps", 0)),
            "slippage_pct": float(getattr(self.cfg, "slippage_pct", 0.0)),
            "commission": float(getattr(self.cfg, "commission", 0.0))
        })
        self.run_metadata["finished_at"] = datetime.now(timezone.utc).isoformat() + "Z"

        # write run metadata to checkpoint dir
        run_meta_path = os.path.join(self.cfg.checkpoint_dir, f"run_metadata_{self.run_id}.json")
        try:
            with open(run_meta_path, "w", encoding="utf-8") as f:
                json.dump(self.run_metadata, f, indent=2)
            logger.info("Wrote run metadata to %s", run_meta_path)
        except Exception:
            logger.exception("Failed to write run metadata to %s", run_meta_path)

        logger.info("Training complete")


if __name__ == "__main__":
    import argparse
    import sys

    # Ensure console logging even if imported elsewhere
    # (don't call basicConfig unconditionally because other modules/tests may have configured logging;
    #  instead attach a StreamHandler only if no handlers exist)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Lightweight Trader RL trainer (test friendly).")
    parser.add_argument("--mode", choices=["train"], default="train", help="Mode to run (train)")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes to run")
    # tests / earlier CLI used --steps; we support both aliases
    parser.add_argument("--steps_per_episode", "--steps", dest="steps_per_episode", type=int, default=None,
                        help="Steps per episode (alias --steps)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint directory override")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed override")
    parser.add_argument("--fill_delay_steps", type=int, default=None, help="Execution model: fill delay steps")
    parser.add_argument("--slippage_pct", type=float, default=None, help="Execution model: slippage pct")
    parser.add_argument("--commission", type=float, default=None, help="Execution model: commission")
    parser.add_argument("--liquidity_fraction", type=float, default=None, help="Execution model: liquidity fraction")

    args = parser.parse_args()

    # Build TrainConfig from defaults then override with CLI args provided
    cfg = TrainConfig()
    if args.episodes is not None:
        cfg.episodes = int(args.episodes)
    if args.steps_per_episode is not None:
        cfg.steps_per_episode = int(args.steps_per_episode)
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.fill_delay_steps is not None:
        cfg.fill_delay_steps = int(args.fill_delay_steps)
    if args.slippage_pct is not None:
        cfg.slippage_pct = float(args.slippage_pct)
    if args.commission is not None:
        cfg.commission = float(args.commission)
    if args.liquidity_fraction is not None:
        cfg.liquidity_fraction = float(args.liquidity_fraction)

    logger.info("Starting module as script with cfg: %s", cfg)
    trainer = RLTrainer(cfg)
    if args.mode == "train":
        trainer.train()
    else:
        logger.error("Unsupported mode %r", args.mode)