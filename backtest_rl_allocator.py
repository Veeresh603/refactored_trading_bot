import pandas as pd
import logging
from datetime import datetime

from strategies.rl_allocator import RLAllocator
from strategies.options_optimizer import OptionsOptimizer
from core import execution_cpp
from core.fees import get_fees   # ✅ Broker fee model

# ----------------------------
# Config
# ----------------------------
CONFIG_FILE = "config.yaml"

import yaml
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

DATA_FILE = "data/nifty_options.csv"
MODEL_PATH = "models/best_allocator_strike_expiry"

slippage_pct = config["backtest"]["slippage_pct"] if "backtest" in config else 0.001
broker_model = config["backtest"]["broker_fees"] if "backtest" in config else "zerodha"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Backtest")

# ----------------------------
# Load Data
# ----------------------------
df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

logger.info(f"📊 Loaded {len(df)} rows of historical data")

# ----------------------------
# Initialize RL Allocator + Optimizer
# ----------------------------
asset_strategies = [("NIFTY", "SMA"), ("NIFTY", "RSI"), ("NIFTY", "RL")]

multi_asset_returns = {s: None for s in asset_strategies}
greek_exposures = {"delta": None, "gamma": None, "vega": None}

rl_alloc = RLAllocator(
    asset_strategies=asset_strategies,
    multi_asset_returns=multi_asset_returns,
    greek_exposures=greek_exposures,
    model_path=MODEL_PATH,
    strikes=[-200, 0, 200],
    expiries=["weekly", "monthly"]
)

opt_config = {"enabled": True, "strike_selection": "dynamic", "expiry_selection": "dynamic", "max_open_positions": 5}
options_optimizer = OptionsOptimizer(opt_config, lot_size=50)

# ----------------------------
# Backtest Loop
# ----------------------------
pnl_curve = []
equity = 2_000_000  # starting balance

for i, row in df.iterrows():
    spot = row["spot"]
    iv = row.get("iv", 0.2)

    # RL decision
    decision = rl_alloc.choose_action()
    strategy, strike_offset, expiry_type = decision["strategy"], decision["strike_offset"], decision["expiry"]

    logger.info(f"{row['date']} → RL chose {strategy}, strike_offset={strike_offset}, expiry={expiry_type}")

    # Build contract
    delta, gamma, vega, theta = execution_cpp.portfolio_greeks(spot)
    greeks = {"delta": delta, "gamma": gamma, "vega": vega}

    contract = options_optimizer.build_option_contract(
        asset="NIFTY",
        spot_price=spot + strike_offset,
        is_call=True,
        iv=iv,
        greeks=greeks
    )

    # --- Apply slippage & broker fees ---
    raw_price = row["close"]
    trade_price = raw_price * (1 + slippage_pct)  # BUY with slippage
    trade_fee = get_fees(broker_model, trade_price, contract["qty"])

    # Simulate execution
    execution_cpp.place_order(
        contract["symbol"], contract["qty"], trade_price, "BUY",
        strike=contract["strike"], sigma=iv, is_call=contract["is_call"],
        expiry_days=(contract["expiry"] - row["date"].date()).days
    )

    # Deduct fees
    equity -= trade_fee

    # Track equity
    status = execution_cpp.account_status(spot)
    equity = status["balance"] + status["total"]
    pnl_curve.append({"date": row["date"], "equity": equity})

# ----------------------------
# Results & Metrics
# ----------------------------
results = pd.DataFrame(pnl_curve)
results.to_csv("backtest_results.csv", index=False)

# Daily returns
results["returns"] = results["equity"].pct_change().fillna(0)

# Metrics
total_return = (results["equity"].iloc[-1] / results["equity"].iloc[0]) - 1
sharpe_ratio = (results["returns"].mean() / results["returns"].std()) * (252 ** 0.5) if results["returns"].std() > 0 else 0

rolling_max = results["equity"].cummax()
drawdown = (results["equity"] - rolling_max) / rolling_max
max_drawdown = drawdown.min()

wins = (results["returns"] > 0).sum()
losses = (results["returns"] <= 0).sum()
win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

# Print summary
logger.info("📊 Backtest Metrics:")
logger.info(f"   Broker Model: {broker_model}")
logger.info(f"   Total Return: {total_return:.2%}")
logger.info(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
logger.info(f"   Max Drawdown: {max_drawdown:.2%}")
logger.info(f"   Win Rate: {win_rate:.2%}")

# Save with metrics
summary = {
    "broker": broker_model,
    "total_return": total_return,
    "sharpe_ratio": sharpe_ratio,
    "max_drawdown": max_drawdown,
    "win_rate": win_rate
}
pd.DataFrame([summary]).to_csv("backtest_summary.csv", index=False)

logger.info("✅ Backtest complete. Results saved to backtest_results.csv and backtest_summary.csv")
