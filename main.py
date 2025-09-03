"""
Main Trading Loop
-----------------
- Supports backtest, paper, and live modes
- Unified strategy evaluation (StrategyEngine)
- Risk managed by AdvancedRiskManager
- Executes via C++ engine (execution_cpp) or broker adapter
"""

import os
import time
import yaml
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

from core import execution_cpp
from core.paper_engine import PaperTradingEngine
from core.utils import send_telegram_message
from core.live_data_manager import LiveDataManager
from core.risk_manager import AdvancedRiskManager

from strategies.rl_allocator import RLAllocator
from strategies.options_optimizer import OptionsOptimizer
from strategies.strategy_engine import StrategyEngine
from strategies.moving_average import MovingAverageStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.mean_reversion import MeanReversionStrategy

# ----------------------------
# Load .env and config.yaml
# ----------------------------
load_dotenv()

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

TRADING_MODE = config["trading"]["mode"]       # backtest | paper | live
BROKER = config["trading"]["broker"]
INITIAL_BALANCE = config["trading"]["initial_balance"]

MARKET_OPEN = datetime.strptime(config["market"]["open"], "%H:%M:%S").time()
MARKET_CLOSE = datetime.strptime(config["market"]["close"], "%H:%M:%S").time()

LOG_LEVEL = getattr(logging, config["logging"]["level"].upper(), logging.INFO)
LOG_FILE = config["logging"]["logfile"]
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger("TradingBot")

# ----------------------------
# Initialize Engines
# ----------------------------
paper_engine = None
live_data_manager = None

if TRADING_MODE == "paper":
    paper_engine = PaperTradingEngine(initial_balance=INITIAL_BALANCE)
    instruments = config["trading"].get("instruments", [256265])  # default NIFTY
    live_data_manager = LiveDataManager(instruments=instruments)
    logger.info("📝 Running in PAPER TRADING mode with real-time feed")

elif TRADING_MODE == "live":
    instruments = config["trading"].get("instruments", [256265])  # default NIFTY
    live_data_manager = LiveDataManager(instruments=instruments)
    logger.info(f"📡 Running in LIVE TRADING mode via {BROKER}")

else:
    logger.info("📊 Running in BACKTEST mode")

# ----------------------------
# Initialize Strategies
# ----------------------------
strategies = [
    MovingAverageStrategy(short_window=20, long_window=50, asset="NIFTY"),
    RSIStrategy(period=14, lower=30, upper=70, asset="NIFTY"),
    MeanReversionStrategy(lookback=20, threshold=0.02, asset="NIFTY"),
]

multi_asset_returns = {("NIFTY", "SMA"): None, ("NIFTY", "RSI"): None, ("NIFTY", "RL"): None}
greek_exposures = {"delta": None, "gamma": None, "vega": None}

rl_alloc = RLAllocator(
    asset_strategies=list(multi_asset_returns.keys()),
    multi_asset_returns=multi_asset_returns,
    greek_exposures=greek_exposures,
    model_path="models/best_allocator_strike_expiry",
    strikes=config["rl_allocator"]["strikes"],
    expiries=config["rl_allocator"]["expiries"],
    window_size=30
)

options_optimizer = OptionsOptimizer(config["options_optimizer"], lot_size=50)
strategy_engine = StrategyEngine(strategies + [rl_alloc])

# ----------------------------
# Risk Manager
# ----------------------------
risk_manager = AdvancedRiskManager(config["risk"])

# ----------------------------
# Circuit Breaker
# ----------------------------
error_count = 0
MAX_ERRORS = 5


# ----------------------------
# Trading Loop
# ----------------------------
def trading_loop():
    global error_count

    logger.info("🚀 Starting trading loop...")
    send_telegram_message("🚀 Trading bot started")

    while True:
        now = datetime.now().time()

        # Market hours check
        if now < MARKET_OPEN or now > MARKET_CLOSE:
            logger.info("⏸️ Market closed, waiting...")
            time.sleep(60)
            continue

        try:
            # ----------------------------
            # Market Data
            # ----------------------------
            if TRADING_MODE in ["paper", "live"]:
                ticks = live_data_manager.get_latest(256265)  # NIFTY
                if not ticks:
                    time.sleep(1)
                    continue
                last_price = ticks["last_price"]
            else:
                last_price = 20000.0  # backtest placeholder

            asset = "NIFTY"

            # ----------------------------
            # Strategy Decision
            # ----------------------------
            market_data = {"asset": asset, "spot": last_price, "history": [last_price]}
            decision = strategy_engine.run(market_data)

            logger.info(f"🤖 StrategyEngine decision: {decision}")

            # ----------------------------
            # Risk Check
            # ----------------------------
            status = execution_cpp.account_status(last_price)
            equity = status["balance"] + status["total"]
            pnl = status["total"]

            safe, reason = risk_manager.check_risk(equity, pnl, abs(50))  # assume fixed lot
            if not safe:
                logger.error(f"❌ Risk triggered: {reason}")
                send_telegram_message(f"❌ Risk triggered: {reason}")
                time.sleep(60)
                continue

            # ----------------------------
            # Build Option Contract
            # ----------------------------
            contract = options_optimizer.evaluate({
                "asset": asset,
                "spot": last_price,
                "strike_offset": decision.get("strike_offset", 0),
                "expiry": decision.get("expiry", "weekly"),
                "greeks": greek_exposures,
                "iv": 0.22
            })

            # ----------------------------
            # Execute Order
            # ----------------------------
            if TRADING_MODE == "live":
                logger.info(f"📡 LIVE ORDER: {contract['symbol']} BUY {last_price}")
                send_telegram_message(f"📡 LIVE ORDER: {contract['symbol']} BUY {last_price}")
                # TODO: integrate broker API here
            elif TRADING_MODE == "paper":
                paper_engine.place_order(
                    contract["symbol"], contract["qty"], last_price, "BUY",
                    strike=contract["strike"], sigma=contract["iv"], is_call=contract["is_call"],
                    expiry_days=(contract["expiry"] - datetime.now().date()).days
                )

            # Update Greeks
            delta, gamma, vega, theta = execution_cpp.portfolio_greeks(last_price)
            greek_exposures.update({"delta": delta, "gamma": gamma, "vega": vega})

            logger.info(f"📈 Greeks Δ={delta:.2f}, Γ={gamma:.4f}, Vega={vega:.2f}, Θ={theta:.2f}")

            # ----------------------------
            # Account Status
            # ----------------------------
            status = execution_cpp.account_status(last_price)
            logger.info(
                f"💰 Balance={status['balance']:.2f}, "
                f"PnL (R={status['realized']:.2f}, U={status['unrealized']:.2f}, T={status['total']:.2f})"
            )
            send_telegram_message(
                f"💰 Balance={status['balance']:.2f}, "
                f"PnL: R={status['realized']:.2f}, U={status['unrealized']:.2f}, T={status['total']:.2f}"
            )

            error_count = 0  # reset errors
            time.sleep(config["trading"].get("poll_interval", 5))

        except Exception as e:
            error_count += 1
            logger.exception(f"❌ Exception in trading loop: {e}")
            send_telegram_message(f"❌ Exception: {e}")
            if error_count >= MAX_ERRORS:
                logger.critical("🛑 Circuit breaker triggered - too many errors")
                send_telegram_message("🛑 Circuit breaker triggered - bot stopped")
                break
            time.sleep(10)


# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    try:
        trading_loop()
    except KeyboardInterrupt:
        logger.info("🛑 Trading bot stopped manually")
        send_telegram_message("🛑 Trading bot stopped manually")
