"""
CLI for Trading Bot
-------------------
Usage Examples:
  python cli.py backtest --strategy sma
  python cli.py train --mode offline --epochs 20
  python cli.py train --mode rl --timesteps 200000
  python cli.py live
"""

import argparse
import logging
import pandas as pd
import os
import subprocess
from glob import glob

from backtesting.backtest import Backtester
from strategies.moving_average import MovingAverageStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.rsi_strategy import RSIStrategy
from core.utils import send_telegram_message

# Training modules
from ai.train_offline import train_offline
from strategies.train_allocator import train_allocator

logger = logging.getLogger("CLI")


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
def run_backtest(strategy_name: str, balance: float, fee: float, sigma: float, expiry: int):
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv("data/historical.csv", parse_dates=["time"])
    if "close" not in df.columns:
        raise ValueError("Historical data must have 'close' column")

    strategy = get_strategy(strategy_name)
    logger.info(f"⚙️ Running backtest with strategy: {strategy_name.upper()}")

    bt = Backtester(strategy=strategy, initial_balance=balance, fee_perc=fee)
    equity, trades_pnl, metrics = bt.run(df, sigma=sigma, expiry_days=expiry)

    print("\n📊 Backtest Results")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.4f}" if isinstance(v, (int, float)) else f"{k:20s}: {v}")

    results_dir = "results"
    equity.to_csv(os.path.join(results_dir, "equity_curve.csv"))
    pd.DataFrame({"pnl": trades_pnl}).to_csv(os.path.join(results_dir, "trades.csv"))

    print(f"\n✅ Results saved in {results_dir}/")

    try:
        send_telegram_message(
            f"📊 Backtest complete with {strategy_name.upper()} strategy.\n"
            f"Total Return={metrics.get('total_return', 0):.2%}"
        )
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


# --------------------------
# Train
# --------------------------
def run_train(train_mode: str, epochs: int, timesteps: int):
    if train_mode == "offline":
        logger.info(f"🚀 Training Offline Models (LSTM/Transformer) for {epochs} epochs")
        train_offline(filepath="data/historical.csv", epochs=epochs, save_dir="models")
        print("✅ Offline training complete. Models saved in models/")

    elif train_mode == "rl":
        logger.info(f"🚀 Training RL Allocator for {timesteps} timesteps")
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
            train_allocator(timesteps=timesteps, save_path=save_path, resume_from=latest_checkpoint)
        else:
            logger.info("🚀 Starting fresh training (no checkpoint found)")
            train_allocator(timesteps=timesteps, save_path=save_path)

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
    parser = argparse.ArgumentParser(description="Trading Bot CLI")
    parser.add_argument("mode", choices=["backtest", "train", "live"], help="Mode to run")

    # Backtest options
    parser.add_argument("--strategy", type=str, default="sma", help="Strategy (sma | rsi | mean)")
    parser.add_argument("--balance", type=float, default=100000, help="Initial balance")
    parser.add_argument("--fee", type=float, default=0.001, help="Fee percentage per trade")
    parser.add_argument("--sigma", type=float, default=0.2, help="Implied volatility")
    parser.add_argument("--expiry", type=int, default=30, help="Expiry in trading days")

    # Train options
    parser.add_argument("--train-mode", type=str, default="offline", help="Training mode (offline | rl)")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for offline training")
    parser.add_argument("--timesteps", type=int, default=100000, help="Timesteps for RL training")

    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest(args.strategy, args.balance, args.fee, args.sigma, args.expiry)
    elif args.mode == "train":
        run_train(args.train_mode, args.epochs, args.timesteps)
    elif args.mode == "live":
        logger.info("🚀 Launching live trading via main.py")
        subprocess.run(["python", "main.py"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
