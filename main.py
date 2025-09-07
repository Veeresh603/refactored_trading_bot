#!/usr/bin/env python3
"""
main.py - unified project runner for backtest / train / paper / live modes

Features:
- CLI: --mode backtest|train|paper|live
- Loads YAML config if provided (fallback to sensible defaults)
- Wires LiveDataManager, execution_engine, strategy, RL trainer
- Defensive imports and fallbacks so the repo can run without external SDKs
- Deterministic seeding and run metadata saved to checkpoint dir

Usage examples:
  python main.py --mode backtest --cfg cfg/backtest.yaml
  python main.py --mode train --episodes 200 --checkpoint-dir checkpoints/exp1
  python main.py --mode paper --replay sample.csv

This file is designed to be copy/paste-ready. Overwrite your existing main.py with this content.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import signal
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Defensive imports (fall back to local implementations we created earlier)
try:
    from core.live_data_manager import LiveDataManager
except Exception:
    LiveDataManager = None

try:
    import core.execution_engine as execution_engine
except Exception:
    execution_engine = None

# Strategy loader: try to load strategies/strategy.py or fallback to a simple SMA strategy
try:
    # expect strategies to expose a Strategy class with on_tick(tick)->List[order_dict]
    from strategies.strategy import Strategy as ProjectStrategy
    HAVE_PROJECT_STRATEGY = True
except Exception:
    ProjectStrategy = None
    HAVE_PROJECT_STRATEGY = False

# RL trainer import
try:
    from ai.train_rl import RLTrainer, TrainConfig
    HAVE_RLTRAINER = True
except Exception:
    RLTrainer = None
    TrainConfig = None
    HAVE_RLTRAINER = False


# -------------------------
# Simple fallback SMA strategy (used if project strategy not present)
# -------------------------
class SimpleSMA:
    """A tiny strategy example to demonstrate wiring.

    Interface:
      - on_tick(tick: dict) -> List[order_dict]
    order_dict: {"symbol": str, "qty": int, "side": "BUY"|"SELL", "order_type": "MARKET"}
    """

    def __init__(self, window_short: int = 5, window_long: int = 20, symbol: str = "TEST", size: int = 1):
        self.window_short = window_short
        self.window_long = window_long
        self.symbol = symbol
        self.size = size
        self.prices: List[float] = []

    def on_tick(self, tick: Dict[str, Any]) -> List[Dict[str, Any]]:
        if tick.get("symbol") != self.symbol:
            return []
        price = float(tick.get("price"))
        self.prices.append(price)
        orders: List[Dict[str, Any]] = []
        if len(self.prices) >= self.window_long:
            short_ma = sum(self.prices[-self.window_short:]) / float(self.window_short)
            long_ma = sum(self.prices[-self.window_long:]) / float(self.window_long)
            # naive crossover
            if short_ma > long_ma:
                orders.append({"symbol": self.symbol, "qty": self.size, "side": "BUY", "order_type": "MARKET"})
            elif short_ma < long_ma:
                orders.append({"symbol": self.symbol, "qty": self.size, "side": "SELL", "order_type": "MARKET"})
        return orders


# -------------------------
# Utilities
# -------------------------

def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_yaml_config(path: str) -> Dict[str, Any]:
    try:
        import yaml

        with open(path, "r") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        logging.getLogger("main").warning("PyYAML not available or failed to load config; using empty config")
        return {}


# -------------------------
# Backtest runner
# -------------------------

def run_backtest(cfg: Dict[str, Any]):
    logger = logging.getLogger("main.backtest")
    logger.info("Starting backtest")

    # instantiate manager and engine
    ldm = LiveDataManager(asset=cfg.get("symbol", "TEST")) if LiveDataManager else None
    if execution_engine and hasattr(execution_engine, "reset_engine"):
        execution_engine.reset_engine(cfg.get("initial_balance", 100000.0))
    else:
        logger.warning("execution_engine not available; backtest cannot run orders")

    # choose strategy
    if HAVE_PROJECT_STRATEGY and ProjectStrategy is not None:
        strat = ProjectStrategy(cfg=cfg)
    else:
        strat = SimpleSMA(window_short=cfg.get("sma_short", 5), window_long=cfg.get("sma_long", 20), symbol=cfg.get("symbol", "TEST"), size=cfg.get("size", 1))

    # replay CSV
    csv_path = cfg.get("replay_csv")
    if not csv_path or not os.path.exists(csv_path):
        logger.error("No replay_csv provided or file not found: %s", csv_path)
        return

    logger.info("Replaying CSV %s", csv_path)
    ticks = ldm.fetch_historical(csv_path, price_col=cfg.get("price_col", "close"))

    for t in ticks:
        # dispatch tick to engine and strategy
        if execution_engine and hasattr(execution_engine, "process_market_tick"):
            execution_engine.process_market_tick(t["symbol"], t["price"], t.get("timestamp"))
        # get orders
        orders = []
        try:
            orders = strat.on_tick(t)
        except Exception:
            logger.exception("Strategy on_tick raised")
        # place orders
        for o in orders:
            try:
                res = execution_engine.place_order(o["symbol"], o["qty"], side=o.get("side", "BUY"), order_type=o.get("order_type", "MARKET"))
                logger.debug("Placed order result: %s", res)
            except Exception:
                logger.exception("Failed to place order")

    # after replay, print summary
    if execution_engine and hasattr(execution_engine, "get_trade_log") and hasattr(execution_engine, "get_positions"):
        trades = execution_engine.get_trade_log()
        pos = execution_engine.get_positions()
        logger.info("Backtest complete — trades=%d positions=%s", len(trades), pos)
        # save summary
        out = cfg.get("output", "backtest_summary.json")
        with open(out, "w") as fh:
            json.dump({"trades": trades, "positions": pos}, fh, default=str, indent=2)
        logger.info("Wrote summary to %s", out)
    else:
        logger.info("Backtest finished (no execution engine info available)")


# -------------------------
# Train runner
# -------------------------

def run_train(args: argparse.Namespace):
    logger = logging.getLogger("main.train")
    if not HAVE_RLTRAINER or RLTrainer is None:
        logger.error("RL trainer not available in this environment")
        return
    cfg = TrainConfig(seed=args.seed, episodes=args.episodes, steps_per_episode=args.steps, checkpoint_dir=args.checkpoint_dir, eval_every=args.eval_every, checkpoint_every=args.checkpoint_every, learning_rate=args.lr, gamma=args.gamma)
    trainer = RLTrainer(cfg)
    trainer.train()


# -------------------------
# Paper / Live runner (scaffold)
# -------------------------

def run_paper(cfg: Dict[str, Any]):
    logger = logging.getLogger("main.paper")
    ldm = LiveDataManager(asset=cfg.get("symbol", "TEST")) if LiveDataManager else None
    if ldm is None:
        logger.error("LiveDataManager not available")
        return

    # Choose strategy
    if HAVE_PROJECT_STRATEGY and ProjectStrategy is not None:
        strat = ProjectStrategy(cfg=cfg)
    else:
        strat = SimpleSMA(window_short=cfg.get("sma_short", 5), window_long=cfg.get("sma_long", 20), symbol=cfg.get("symbol", "TEST"), size=cfg.get("size", 1))

    # subscribe strategy to ticks via callback
    def on_tick(tick):
        try:
            orders = strat.on_tick(tick)
            for o in orders:
                try:
                    if execution_engine and hasattr(execution_engine, "place_order"):
                        res = execution_engine.place_order(o["symbol"], o["qty"], side=o.get("side", "BUY"), order_type=o.get("order_type", "MARKET"))
                        logger.debug("Paper order result: %s", res)
                except Exception:
                    logger.exception("Paper mode: failed to place order")
        except Exception:
            logger.exception("Strategy on_tick failed in paper mode")

    ldm.subscribe(cfg.get("symbol", "TEST"), on_tick)

    # start live mode (best-effort)
    ldm.start_live(connect_info=cfg.get("connect_info"), symbols=[cfg.get("symbol")])
    logger.info("Paper mode running — press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Stopping paper mode")
        ldm.stop()


def run_live(cfg: Dict[str, Any]):
    logger = logging.getLogger("main.live")
    logger.warning("Live mode scaffold: ensure you understand the risks before connecting real accounts")
    run_paper(cfg)  # for now paper and live share mechanics in scaffold


# -------------------------
# Signal handling & metadata
# -------------------------

def _write_run_metadata(base_dir: str, meta: Dict[str, Any]):
    try:
        os.makedirs(base_dir, exist_ok=True)
        fn = os.path.join(base_dir, f"run_meta_{int(time.time())}.json")
        with open(fn, "w") as fh:
            json.dump(meta, fh, default=str, indent=2)
    except Exception:
        logging.getLogger("main").exception("Failed to write run metadata")


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, required=True, choices=["backtest", "train", "paper", "live"], help="Mode to run")
    p.add_argument("--cfg", type=str, default=None, help="YAML config file path for backtest/paper/live")

    # train options
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)

    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("main")

    cfg: Dict[str, Any] = {}
    if args.cfg:
        cfg = load_yaml_config(args.cfg)
    # override simple settings from CLI
    cfg["seed"] = args.seed
    cfg["symbol"] = cfg.get("symbol", "TEST")

    # write run metadata directory
    meta_dir = os.path.join(args.checkpoint_dir, "runs")
    meta = {"mode": args.mode, "cfg": cfg, "timestamp": datetime.now(timezone).isoformat()}
    _write_run_metadata(meta_dir, meta)

    # dispatch modes
    if args.mode == "backtest":
        run_backtest(cfg)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "paper":
        run_paper(cfg)
    elif args.mode == "live":
        run_live(cfg)
    else:
        logger.error("Unknown mode: %s", args.mode)


if __name__ == "__main__":
    main()
