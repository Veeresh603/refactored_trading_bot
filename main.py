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
from strategies.rl_allocator import RLAllocator
from strategies.options_optimizer import OptionsOptimizer

# ----------------------------
# Load .env and config.yaml
# ----------------------------
load_dotenv()

status = execution_cpp.account_status(20000)


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


config = load_config()

TRADING_MODE = config["trading"]["mode"]       # "backtest" | "paper" | "live"
BROKER = config["trading"]["broker"]
INITIAL_BALANCE = config["trading"]["initial_balance"]

MARKET_OPEN = datetime.strptime(config["market"]["open"], "%H:%M:%S").time()
MARKET_CLOSE = datetime.strptime(config["market"]["close"], "%H:%M:%S").time()

# Risk params (will be refreshed dynamically)
RISK_CONFIG = config["risk"]

OPTIMIZER_CONFIG = config["options_optimizer"]
RL_CONFIG = config["rl_allocator"]
STRIKES = RL_CONFIG["strikes"]
EXPIRIES = RL_CONFIG["expiries"]

LOG_LEVEL = getattr(logging, config["logging"]["level"].upper(), logging.INFO)
LOG_FILE = config["logging"]["logfile"]

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# ----------------------------
# Logger setup
# ----------------------------
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE,  encoding="utf-8"), logging.StreamHandler()]
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
# Initialize RL Allocator & Optimizer
# ----------------------------
multi_asset_returns = {
    ("NIFTY", "SMA"): None,
    ("NIFTY", "RSI"): None,
    ("NIFTY", "RL"): None,
}
greek_exposures = {"delta": None, "gamma": None, "vega": None}

rl_alloc = RLAllocator(
    asset_strategies=list(multi_asset_returns.keys()),
    multi_asset_returns=multi_asset_returns,
    greek_exposures=greek_exposures,
    model_path="models/best_allocator_strike_expiry",
    strikes=STRIKES,
    expiries=EXPIRIES,
    window_size=30
)

options_optimizer = OptionsOptimizer(OPTIMIZER_CONFIG, lot_size=50)

# ----------------------------
# Risk tracking state
# ----------------------------
daily_realized = 0
peak_equity = INITIAL_BALANCE
cool_off_until = None
consecutive_losses = 0


# ----------------------------
# Refresh Risk Parameters
# ----------------------------
def load_risk_params():
    try:
        cfg = load_config()
        return cfg.get("risk", {})
    except Exception as e:
        logger.error(f"❌ Failed to reload risk params: {e}")
        return RISK_CONFIG


# ----------------------------
# Trading loop
# ----------------------------
def trading_loop():
    global peak_equity, cool_off_until, consecutive_losses, daily_realized, RISK_CONFIG

    logger.info("🚀 Starting trading loop...")
    send_telegram_message("🚀 Trading bot started")

    last_reload = time.time()
    refresh_interval = 300  # 5 min

    while True:
        now = datetime.now().time()

        # --- Risk param auto-refresh ---
        if time.time() - last_reload > refresh_interval:
            new_risk = load_risk_params()
            if new_risk != RISK_CONFIG:
                RISK_CONFIG = new_risk
                logger.info(f"🔄 Risk parameters updated: {RISK_CONFIG}")
                send_telegram_message(f"⚡ Risk parameters reloaded: {RISK_CONFIG}")
            last_reload = time.time()

        MAX_POSITION_SIZE = RISK_CONFIG["max_position_size"]
        MAX_DAILY_LOSS = RISK_CONFIG["max_daily_loss"]
        MAX_DRAWDOWN = RISK_CONFIG["max_drawdown"]
        STOP_ON_LOSS = RISK_CONFIG["stop_on_loss"]
        COOL_OFF_MINUTES = RISK_CONFIG["cool_off_minutes"]
        MAX_CONSECUTIVE_LOSSES = RISK_CONFIG["max_consecutive_losses"]

        # Market closed
        if now < MARKET_OPEN or now > MARKET_CLOSE:
            logger.info("⏸️ Market closed, waiting...")
            time.sleep(60)
            continue

        # Cool-off check
        if cool_off_until and datetime.now() < cool_off_until:
            logger.warning(f"⏸️ Cool-off active until {cool_off_until.strftime('%H:%M:%S')}")
            time.sleep(30)
            continue
        else:
            cool_off_until = None

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
            # Backtest fallback (replace with historical feed)
            last_price = 20000.0

        asset = "NIFTY"

        # ----------------------------
        # RL Allocator decision
        # ----------------------------
        decision = rl_alloc.choose_action()
        chosen_strategy, strike_offset, expiry_type = (
            decision["strategy"],
            decision["strike_offset"],
            decision["expiry"],
        )

        logger.info(f"🤖 RL chose {chosen_strategy} with strike_offset={strike_offset}, expiry={expiry_type}")

        # Account status before trade
        status = execution_cpp.account_status(last_price)
        equity = status["balance"] + status["total"]

        # Track drawdown
        peak_equity = max(peak_equity, equity)
        drawdown = peak_equity - equity

        # --- Risk Checks ---
        if 50 > MAX_POSITION_SIZE:  # Example fixed qty
            logger.warning("⚠️ Position size exceeds risk limit, skipping order")
            continue

        if abs(status["realized"]) > MAX_DAILY_LOSS:
            logger.error("❌ Max daily loss breached!")
            send_telegram_message("❌ Max Daily Loss Breached!")
            if STOP_ON_LOSS:
                logger.error("🛑 Trading stopped for the day")
                send_telegram_message("🛑 Trading stopped for the day")
                break
            else:
                cool_off_until = datetime.now() + timedelta(minutes=COOL_OFF_MINUTES)
                continue

        if drawdown > MAX_DRAWDOWN:
            logger.error("❌ Max drawdown breached!")
            send_telegram_message("❌ Max Drawdown Breached!")
            if STOP_ON_LOSS:
                logger.error("🛑 Trading stopped for the day")
                send_telegram_message("🛑 Trading stopped for the day")
                break
            else:
                cool_off_until = datetime.now() + timedelta(minutes=COOL_OFF_MINUTES)
                continue

        # ----------------------------
        # Optimizer builds exact option contract
        # ----------------------------
        delta, gamma, vega, theta = execution_cpp.portfolio_greeks(last_price)
        greeks = {"delta": delta, "gamma": gamma, "vega": vega}

        iv = 0.22  # TODO: replace with real IV feed

        contract = options_optimizer.build_option_contract(
            asset=asset,
            spot_price=last_price + strike_offset,  # RL offset applied
            is_call=True,
            iv=iv,
            greeks=greeks,
        )

        # ----------------------------
        # Place order
        # ----------------------------
        if TRADING_MODE == "live":
            logger.info(f"📡 LIVE ORDER: {contract['symbol']} BUY {last_price}")
            send_telegram_message(f"📡 LIVE ORDER: {contract['symbol']} BUY {last_price}")
            # TODO: Add broker integration
        elif TRADING_MODE == "paper":
            paper_engine.place_order(
                contract["symbol"], contract["qty"], last_price, "BUY",
                strike=contract["strike"], sigma=iv, is_call=contract["is_call"],
                expiry_days=(contract["expiry"] - datetime.now().date()).days
            )

        # Greeks update
        delta, gamma, vega, theta = execution_cpp.portfolio_greeks(last_price)
        logger.info(f"📈 Greeks Δ={delta:.2f}, Γ={gamma:.4f}, Vega={vega:.2f}, Θ={theta:.2f}")

        # Account status after trade
        prev_realized = status["realized"]
        status = execution_cpp.account_status(last_price)

        if status["realized"] < prev_realized:
            consecutive_losses += 1
            logger.warning(f"⚠️ Consecutive loss detected ({consecutive_losses} in a row)")
            if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                logger.error("❌ Max consecutive losses reached!")
                send_telegram_message("❌ Max consecutive losses reached!")
                if STOP_ON_LOSS:
                    break
                else:
                    cool_off_until = datetime.now() + timedelta(minutes=COOL_OFF_MINUTES)
                    consecutive_losses = 0
                    continue
        else:
            consecutive_losses = 0

        logger.info(
            f"💰 Balance={status['balance']:.2f}, "
            f"Margin={status['margin_used']:.2f}, "
            f"PnL (R={status['realized']:.2f}, U={status['unrealized']:.2f}, T={status['total']:.2f})"
        )
        send_telegram_message(
            f"💰 Balance={status['balance']:.2f}, "
            f"PnL: R={status['realized']:.2f}, U={status['unrealized']:.2f}, T={status['total']:.2f}"
        )

        time.sleep(config["trading"].get("poll_interval", 5))


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    try:
        trading_loop()
    except KeyboardInterrupt:
        logger.info("🛑 Trading bot stopped manually")
        send_telegram_message("🛑 Trading bot stopped manually")
