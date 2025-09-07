"""
CLI for Trading Bot
-------------------
Usage Examples:
  python cli_entry.py backtest --strategy sma
  python cli_entry.py train --mode offline --epochs 20
  python cli_entry.py train --mode rl --timesteps 200000 --seed 42
  python cli_entry.py live
"""

import argparse
import logging
import pandas as pd
import os
import subprocess
from glob import glob
from typing import Optional

from backtesting.backtest import Backtester

# Optional OrderBookSampler — use if available
try:
    from backtesting.orderbook_sampler import OrderBookSampler
except Exception:
    OrderBookSampler = None

# Strategies (keep your existing imports)
from strategies.moving_average import MovingAverageStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.rsi_strategy import RSIStrategy
from core.utils import send_telegram_message

# Training modules (these may be optional in your project)
try:
    from ai.train_offline import train_offline
except Exception:
    train_offline = None

try:
    from strategies.train_allocator import train_allocator
except Exception:
    train_allocator = None

logger = logging.getLogger("CLI")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --------------------------
# Strategy Factory
# --------------------------
def get_strategy(name: str):
    name = name.lower()
    if name == "sma":
        return MovingAverageStrategy(short_window=20, long_window=50)
    elif name == "mean":
        return MeanReversionStrategy(lookback=20, threshold=0.02)
    elif name == "rsi":
        return RSIStrategy(period=14, lower=30, upper=70)
    else:
        raise ValueError(f"❌ Unknown strategy: {name}")


# --------------------------
# Backtest
# --------------------------
def run_backtest(
    strategy_name: str,
    balance: float,
    commission: float,
    slippage: float,
    position_size: float,
    liquidity_fraction: float,
    fill_delay: int,
    seed: Optional[int],
    price_path: str = "data/historical.csv",
):
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(price_path):
        raise FileNotFoundError(f"Price CSV not found: {price_path}")

    df = pd.read_csv(price_path, parse_dates=["time"])
    if "close" not in df.columns:
        raise ValueError("Historical data must have 'close' column")

    strategy = get_strategy(strategy_name)
    logger.info(f"⚙️ Running backtest with strategy: {strategy_name.upper()}")

    # Build optional OrderBookSampler if available and user provided a seed
    obs_sampler = None
    if OrderBookSampler is not None and seed is not None:
        try:
            obs_sampler = OrderBookSampler(df=df, liquidity_fraction=liquidity_fraction, seed=int(seed))
            logger.info("Using OrderBookSampler for execution simulation (seed=%s)", seed)
        except Exception:
            logger.warning("Failed to instantiate OrderBookSampler; proceeding without it")

    bt = Backtester(
        strategy=strategy,
        initial_balance=balance,
        fill_delay_steps=fill_delay,
        slippage_pct=slippage,
        commission=commission,
        position_size=position_size,
        liquidity_fraction=liquidity_fraction,
        orderbook_sampler=obs_sampler,
    )

    # Run backtest (Backtester.run signature: df, price_col='close', ...)
    equity, trades_df, metrics = bt.run(df)

    print("\n📊 Backtest Results")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"{k:20s}: {v:.6f}")
        else:
            print(f"{k:20s}: {v}")

    results_dir = "results"
    equity.to_csv(os.path.join(results_dir, "equity_curve.csv"))
    trades_df.to_csv(os.path.join(results_dir, "trades.csv"), index=False)

    print(f"\n✅ Results saved in {results_dir}/")

    try:
        send_telegram_message(
            f"📊 Backtest complete ({strategy_name.upper()}). Total Return={metrics.get('final_return', 0):.2%}"
        )
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


# --------------------------
# Train
# --------------------------
def run_train(train_mode: str, epochs: int, timesteps: int, seed: Optional[int]):
    if train_mode == "offline":
        if train_offline is None:
            raise RuntimeError("Offline training module not available (ai.train_offline)")
        logger.info(f"🚀 Training Offline Models (LSTM/Transformer) for {epochs} epochs")
        train_offline(filepath="data/historical.csv", epochs=epochs, save_dir="models")
        print("✅ Offline training complete. Models saved in models/")

    elif train_mode == "rl":
        if train_allocator is None:
            raise RuntimeError("RL allocator training module not available (strategies.train_allocator)")
        logger.info(f"🚀 Training RL Allocator for {timesteps} timesteps (seed={seed})")
        save_path = "models/best_allocator_strike_expiry"
        checkpoint_dir = os.path.join(save_path, "checkpoints")

        # Check for checkpoint to resume
        latest_checkpoint = None
        if os.path.exists(checkpoint_dir):
            checkpoints = glob(os.path.join(checkpoint_dir, "*_steps.zip"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)

        if latest_checkpoint:
            logger.info(f"🔄 Resuming training from checkpoint: {latest_checkpoint}")
            train_allocator(timesteps=timesteps, save_path=save_path, resume_from=latest_checkpoint, seed=seed)
        else:
            logger.info("🚀 Starting fresh training (no checkpoint found)")
            train_allocator(timesteps=timesteps, save_path=save_path, seed=seed)

        print(f"✅ RL Allocator training complete.")
        print(f"📂 Final Model: {save_path}")
        print(f"📂 Best Checkpoints: {checkpoint_dir}")

        try:
            send_telegram_message("✅ RL Allocator training complete.")
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")

    else:
        raise ValueError(f"❌ Unknown training mode: {train_mode}")


# --------------------------
# CLI Entry Point
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Trading Bot CLI (simple)")
    parser.add_argument("mode", choices=["backtest", "train", "live"], help="Mode to run")

    # Backtest options
    parser.add_argument("--strategy", type=str, default="sma", help="Strategy (sma | rsi | mean)")
    parser.add_argument("--balance", type=float, default=100000.0, help="Initial balance")
    parser.add_argument("--fee", type=float, default=0.0, help="Commission per trade (absolute amount)")
    parser.add_argument("--slippage", type=float, default=0.0, help="Slippage percent (e.g. 0.001)")
    parser.add_argument("--position-size", type=float, default=1.0, help="Units per trade")
    parser.add_argument("--liquidity-fraction", type=float, default=0.1, help="Fraction of bar volume available")
    parser.add_argument("--fill-delay", type=int, default=1, help="Fill delay in bars (integer)")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed (affects orderbook sampler / training reproducibility)")

    # Train options
    parser.add_argument("--train-mode", type=str, default="offline", help="Training mode (offline | rl)")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for offline training")
    parser.add_argument("--timesteps", type=int, default=100000, help="Timesteps for RL training")

    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest(
            strategy_name=args.strategy,
            balance=args.balance,
            commission=args.fee,
            slippage=args.slippage,
            position_size=args.position_size,
            liquidity_fraction=args.liquidity_fraction,
            fill_delay=args.fill_delay,
            seed=args.seed,
        )
    elif args.mode == "train":
        run_train(train_mode=args.train_mode, epochs=args.epochs, timesteps=args.timesteps, seed=args.seed)
    elif args.mode == "live":
        logger.info("🚀 Launching live trading via main.py")
        subprocess.run(["python", "main.py"])


if __name__ == "__main__":
    main()
