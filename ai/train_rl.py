"""
ai/train_rl.py

Trainer that is aligned with the provided ai/rl_agent.py API:

- RLAgent(obs_dim, act_dim, lr=..., gamma=...)
- agent.select_action(obs[, deterministic=False]) -> int
- agent.store_transition(obs, action, reward)
- agent.update()
- agent.save(path) / load(path)

Features:
- If data/historical.csv is missing, automatically invokes scripts/generate_synthetic_data.py
  to create a small CSV so training can start immediately.
- Defensive imports & helpful logging for troubleshooting missing modules.
- Saves a simple metrics JSON and the trained agent (pickle).
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------
# Logging
# ---------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_rl")

# ---------------------------------------
# Paths & constants
# ---------------------------------------
DATA_PATH = Path("data/historical.csv")
GENERATOR_SCRIPT = Path("scripts/generate_synthetic_data.py")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_data_exists(out_path: Path = DATA_PATH, days: float = 1.0, freq: str = "5T", seed: int = 42) -> bool:
    """
    Ensure training CSV exists; if not run the generator script.
    Returns True if file exists after call.
    """
    if out_path.exists():
        logger.info("Data file found: %s", out_path)
        return True

    if not GENERATOR_SCRIPT.exists():
        logger.warning("Synthetic generator not found at '%s' — please create %s manually.", GENERATOR_SCRIPT, out_path)
        return False

    logger.info("data/historical.csv missing — running synthetic generator: %s", GENERATOR_SCRIPT)
    cmd = [
        sys.executable,
        str(GENERATOR_SCRIPT),
        "--out", str(out_path),
        "--days", str(days),
        "--freq", freq,
        "--seed", str(seed),
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        logger.exception("Synthetic generator timed out.")
        return False

    if proc.returncode != 0:
        logger.error("Synthetic generator failed (rc=%s). stdout:\n%s\nstderr:\n%s", proc.returncode, proc.stdout, proc.stderr)
        return False

    logger.info("Generator finished. stdout:\n%s", proc.stdout.strip())
    time.sleep(0.1)
    exists = out_path.exists()
    if not exists:
        logger.error("Generator finished but file missing at %s", out_path)
    else:
        logger.info("Created data file: %s (size=%d bytes)", out_path, out_path.stat().st_size)
    return exists


def train_rl(
    episodes: int = 50,
    steps_per_episode: int = 200,
    generate_if_missing: bool = True,
    generator_days: float = 1.0,
    generator_freq: str = "5T",
    seed: Optional[int] = 42,
):
    """
    Train RL agent using the project's TradingEnv and the provided RLAgent API.
    """

    # ensure data exists
    if generate_if_missing:
        ok = ensure_data_exists(out_path=DATA_PATH, days=generator_days, freq=generator_freq, seed=seed)
        if not ok:
            logger.error("Training data missing and generator failed. Aborting.")
            return None
    elif not DATA_PATH.exists():
        logger.error("Training data %s not found and generation disabled.", DATA_PATH)
        return None

    # import env & agent with helpful errors
    try:
        from envs.trading_env import TradingEnv
    except Exception as e:
        logger.exception("Failed to import TradingEnv from envs/trading_env.py: %s", e)
        raise

    try:
        from ai.models.rl_agent import RLAgent
    except Exception as e:
        logger.exception("Failed to import RLAgent from ai/rl_agent.py: %s", e)
        raise

    # optional metrics
    try:
        from backtesting.metrics import compute_metrics
    except Exception:
        compute_metrics = None
        logger.info("Optional metrics module backtesting.metrics not found — continuing without it.")

    # instantiate env - try a couple forms
    try:
        env = TradingEnv(data_path=str(DATA_PATH))
    except TypeError:
        env = TradingEnv()
    except Exception as e:
        logger.exception("Failed to instantiate TradingEnv: %s", e)
        raise

    # Discover obs/action dims defensively
    if hasattr(env, "observation_space") and hasattr(env.observation_space, "shape"):
        obs_dim = int(env.observation_space.shape[0])
    elif hasattr(env, "obs_dim"):
        obs_dim = int(env.obs_dim)
    else:
        # fallback: infer from a reset observation
        try:
            sample_obs = env.reset()
            import numpy as _np
            sample_arr = _np.asarray(sample_obs)
            obs_dim = int(sample_arr.shape[0]) if sample_arr.ndim >= 1 else 1
            logger.info("Inferred obs_dim=%s from env.reset()", obs_dim)
        except Exception:
            obs_dim = 10
            logger.warning("Could not infer obs_dim; using default obs_dim=%d", obs_dim)

    if hasattr(env, "action_space") and hasattr(env.action_space, "n"):
        act_dim = int(env.action_space.n)
    elif hasattr(env, "act_dim"):
        act_dim = int(env.act_dim)
    else:
        # fallback default: 3 discrete actions (sell, neutral, buy)
        act_dim = 3
        logger.warning("Could not infer act_dim; using default act_dim=%d", act_dim)

    logger.info("Environment dims: obs_dim=%s act_dim=%s", obs_dim, act_dim)

    # Instantiate agent using the exact RLAgent API you provided
    agent = RLAgent(obs_dim=obs_dim, act_dim=act_dim, lr=3e-4, gamma=0.99, seed=seed)
    logger.info("Initialized RLAgent with lr=%s gamma=%s", agent.lr, agent.gamma)

    # Training loop
    metrics_all = []
    for ep in range(1, episodes + 1):
        obs = env.reset()
        ep_reward = 0.0
        done = False

        # If env.reset returns dict-like observation (e.g., {"obs": ...}), try to extract
        if isinstance(obs, dict) and "obs" in obs:
            obs = obs["obs"]

        for step in range(steps_per_episode):
            # ensure 1-D numpy array for policy
            try:
                import numpy as np
                obs_arr = np.asarray(obs, dtype=float).reshape(-1)  # shape (obs_dim,)
            except Exception:
                obs_arr = obs  # pass-through; agent will error if incompatible

            # ask agent for action using select_action
            try:
                action = agent.select_action(obs_arr, deterministic=False)
            except Exception as e:
                logger.exception("agent.select_action failed at ep=%d step=%d: %s", ep, step, e)
                raise

            # step the env
            try:
                nxt_obs, reward, done, info = env.step(action)
            except Exception as e:
                logger.exception("env.step failed at ep=%d step=%d: %s", ep, step, e)
                raise

            # store transition with the exact signature expected
            try:
                agent.store_transition(obs_arr, int(action), float(reward))
            except Exception as e:
                logger.exception("agent.store_transition failed: %s", e)
                raise

            ep_reward += float(reward)
            obs = nxt_obs

            if done:
                break

        # After episode, call agent.update() (REINFORCE uses whole episode)
        try:
            agent.update()
        except Exception as e:
            logger.exception("agent.update() failed at end of episode %d: %s", ep, e)
            raise

        logger.info("Episode %d/%d finished: reward=%.4f steps=%d", ep, episodes, ep_reward, step + 1)

        # optional metrics
        if compute_metrics is not None:
            try:
                m = compute_metrics(env)
                metrics_all.append(m)
                logger.info("Episode %d metrics: %s", ep, m)
            except Exception:
                logger.exception("compute_metrics failed for episode %d", ep)

        # periodic checkpointing of agent
        if ep % 10 == 0 or ep == episodes:
            ckpt_path = MODELS_DIR / f"rl_agent_ep{ep}.pkl"
            try:
                agent.save(str(ckpt_path))
                logger.info("Saved agent checkpoint: %s", ckpt_path)
            except Exception:
                logger.exception("Failed to save agent checkpoint at episode %d", ep)

    # final save
    final_path = MODELS_DIR / "rl_agent_final.pkl"
    try:
        agent.save(str(final_path))
        logger.info("Saved final agent to %s", final_path)
    except Exception:
        logger.exception("Failed to save final agent")

    # write metrics summary
    if metrics_all:
        out_metrics = MODELS_DIR / "train_metrics.json"
        with out_metrics.open("w") as fh:
            json.dump(metrics_all, fh, indent=2, default=str)
        logger.info("Saved metrics to %s", out_metrics)
    else:
        logger.info("No episode metrics collected (compute_metrics not present or returned nothing).")

    return metrics_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps-per-episode", type=int, default=200)
    parser.add_argument("--no-generate", action="store_true", help="Do not generate synthetic data if missing")
    parser.add_argument("--generate-days", type=float, default=1.0)
    parser.add_argument("--generate-freq", default="5T")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        metrics = train_rl(
            episodes=args.episodes,
            steps_per_episode=args.steps_per_episode,
            generate_if_missing=(not args.no_generate),
            generator_days=args.generate_days,
            generator_freq=args.generate_freq,
            seed=args.seed,
        )
    except Exception as e:
        logger.exception("Training aborted with exception: %s", e)
        sys.exit(1)

    logger.info("Training completed. Metrics collected: %d", len(metrics) if metrics else 0)


if __name__ == "__main__":
    main()
