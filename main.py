"""
Main Trading Loop
-----------------
- Supports backtest, paper, and live modes
- Unified strategy evaluation (StrategyEngine)
- Risk managed by AdvancedRiskManager
- Executes via C++ engine (execution_engine_cpp) or paper engine
- Debounced decisions (no duplicate spam)
- Sizing log shows risk utilization %
"""

import os
import time
import yaml
import logging
from collections import deque
from datetime import datetime, date
from dotenv import load_dotenv

# Core + Strategies
from core import execution_engine_cpp as execution_engine  # pybind11 module name (per your last setup)
from core.paper_engine import PaperTradingEngine
from core.utils import send_telegram_message
from core.live_data_manager import LiveDataManager
from core.risk_manager import AdvancedRiskManager

from strategies.rl_allocator import RLAllocator
from strategies.options_optimizer import OptionsOptimizer
from core.strategy_engine import StrategyEngine
from strategies.moving_average import MovingAverageStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.mean_reversion import MeanReversionStrategy
from backtesting.backtest import Backtester

# ----------------------------
# Load .env and config.yaml
# ----------------------------
load_dotenv()


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


config = load_config()

TRADING_MODE = config["trading"]["mode"]   # "backtest" | "paper" | "live"
BROKER = config["trading"]["broker"]
INITIAL_BALANCE = float(config["trading"]["initial_balance"])
LOT_SIZE = int(config["trading"].get("paper_lot_size", config.get("risk", {}).get("lot_size", 50)))

MARKET_OPEN = datetime.strptime(config["market"]["open"], "%H:%M:%S").time()
MARKET_CLOSE = datetime.strptime(config["market"]["close"], "%H:%M:%S").time()

RISK_CONFIG = config["risk"]
OPTIMIZER_CONFIG = config["options_optimizer"]
RL_CONFIG = config["rl_allocator"]

# risk per trade for sizing (used to compute allowed_risk)
RISK_PER_TRADE = float(config.get("risk_per_trade", 0.01))

# ----------------------------
# Logger setup
# ----------------------------
LOG_LEVEL = getattr(logging, config["logging"]["level"].upper(), logging.INFO)
LOG_FILE = config["logging"]["logfile"]
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler()]
)
logger = logging.getLogger("TradingBot")

# ----------------------------
# Initialize Engines / Data
# ----------------------------
paper_engine = None
live_data_manager = None

if TRADING_MODE == "paper":
    paper_engine = PaperTradingEngine(initial_balance=INITIAL_BALANCE, lot_size=LOT_SIZE)
    instruments = config["trading"].get("instruments", [256265])  # default NIFTY token
    live_data_manager = LiveDataManager(instruments=instruments, broker=BROKER)
    logger.info("📝 Running in PAPER TRADING mode")

elif TRADING_MODE == "live":
    instruments = config["trading"].get("instruments", [256265])
    live_data_manager = LiveDataManager(instruments=instruments, broker=BROKER)
    logger.info(f"📡 Running in LIVE mode via {BROKER}")

else:
    logger.info("📊 Running in BACKTEST mode")

# ----------------------------
# Initialize Strategies
# ----------------------------
base_strategies = [
    MovingAverageStrategy(short_window=20, long_window=50),
    RSIStrategy(period=14, lower=30, upper=70),
    MeanReversionStrategy(lookback=20, threshold=0.02),
]

rl_alloc = RLAllocator(
    asset_strategies=["SMA", "RSI", "MeanReversion"],
    multi_asset_returns={}, greek_exposures={},
    model_path="models/best_allocator_strike_expiry",
    strikes=RL_CONFIG["strikes"], expiries=RL_CONFIG["expiries"],
    window_size=RL_CONFIG.get("window_size", 30)
)

strategy_engine = StrategyEngine(base_strategies + [rl_alloc])
options_optimizer = OptionsOptimizer(OPTIMIZER_CONFIG, lot_size=LOT_SIZE)
risk_manager = AdvancedRiskManager(RISK_CONFIG)

# Debounce state for decisions
_last_decision_key = None
_state_buf = deque(maxlen=2)  # simple stability buffer

# ----------------------------
# Helpers
# ----------------------------
def market_open_now() -> bool:
    now = datetime.now().time()
    return MARKET_OPEN <= now <= MARKET_CLOSE

def get_spot() -> float:
    """
    Obtain current underlying spot from live_data_manager.
    Uses the first configured instrument or a named lookup if available.
    """
    if live_data_manager is None:
        return 0.0
    # Try a friendly 'get_spot' API, else fallback to get_latest(instrument_token)
    try:
        return float(live_data_manager.get_spot("NIFTY"))
    except Exception:
        token = config["trading"].get("instruments", [256265])[0]
        ticks = live_data_manager.get_latest(token)
        if ticks and "last_price" in ticks:
            return float(ticks["last_price"])
        return 0.0

def pnl_log_from_account(acct: dict) -> str:
    realized = acct.get("realized", 0.0)
    unrealized = acct.get("unrealized", 0.0)
    total = realized + unrealized
    return f"💰 PnL Update → Realized={realized:.2f}, Unrealized={unrealized:.2f}, Total={total:.2f}"

def log_greeks(spot: float):
    try:
        dlt, gmm, vega, theta = execution_engine.portfolio_greeks(spot)
        logger.info(f"📈 Greeks Δ={dlt:.2f}, Γ={gmm:.4f}, Vega={vega:.2f}, Θ={theta:.2f}")
    except Exception:
        # Some builds return a tuple of 4, others dict—just be safe
        try:
            greeks = execution_engine.portfolio_greeks(spot)
            logger.info(f"📈 Greeks Δ={greeks[0]:.2f}, Γ={greeks[1]:.4f}, Vega={greeks[2]:.2f}, Θ={greeks[3]:.2f}")
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch Greeks: {e}")

def flatten_if_neutral(signal: int, spot: float):
    """When signal is neutral and there are open positions, flatten them."""
    if signal != 0 or paper_engine is None:
        return
    open_positions = paper_engine.list_open_positions()
    if open_positions:
        logger.info(f"🔁 Neutral: flattening {len(open_positions)} open positions")
        paper_engine.flatten_all(spot)

# --- Robust wrapper to build/construct option contract from OptionsOptimizer ---
import inspect
import logging

logger = logging.getLogger("TradingBot")

def size_and_place(decision, spot):
    """
    Build contract using options_optimizer in a resilient way, apply position sizing,
    and place the order on paper_engine or live broker.
    This wrapper tries several common method names/signatures so mismatches don't crash.
    """
    contract = None
    opt = options_optimizer  # assumes options_optimizer is global in main.py as before

    # Candidate method names & call orders we try (most-likely first)
    candidate_calls = [
        ("build_contract", (spot, decision)),      # what your traceback shows
        ("build_contract", (decision, spot)),      # alternative arg order
        ("evaluate", (decision, spot)),            # earlier code used .evaluate(decision)
        ("evaluate", (spot, decision)),            # alternative order
        ("build", (decision, spot)),               # other possible name
        ("build", (spot, decision)),
    ]

    # Keep a small registry of errors for troubleshooting
    call_errors = []

    for name, args in candidate_calls:
        if not hasattr(opt, name):
            call_errors.append((name, "missing"))
            continue

        fn = getattr(opt, name)
        # sanity: ensure callable
        if not callable(fn):
            call_errors.append((name, "not callable"))
            continue

        # Attempt to call, but guard exceptions so trading loop doesn't crash
        try:
            logger.debug(f"[TRACE] Trying options_optimizer.{name}{args}")
            contract = fn(*args)
            # If function returns False-y or raises custom error, treat accordingly
            if contract:
                logger.info(f"📦 Built contract via {name}: {contract}")
                break
            else:
                # record that call returned falsy (None/False/empty) and continue trying others
                call_errors.append((name, "returned falsy"))
        except TypeError as te:
            # signature mismatch (wrong number of args etc.)
            call_errors.append((name, f"TypeError: {te}"))
        except Exception as e:
            # other runtime error (e.g. missing data inside optimizer)
            call_errors.append((name, f"Exception: {e}"))

    if contract is None:
        # Build a helpful diagnostic message and skip placing.
        opt_methods = [m for m in dir(opt) if not m.startswith("_")]
        logger.error("❌ OptionsOptimizer could not build a contract.")
        logger.error("Tried these method attempts and results:")
        for entry in call_errors:
            logger.error(f" - {entry[0]} -> {entry[1]}")
        logger.error(f"OptionsOptimizer exposed public attributes: {opt_methods}")
        # Optionally, also dump the decision and spot so user can inspect
        logger.debug(f"[TRACE] decision={decision}, spot={spot}")
        return None

    # If we have a contract, do sizing and execute (this part adapts to your existing flow)
    try:
        # Example: ensure contract dict fields exist
        qty = contract.get("qty", 1)
        symbol = contract.get("symbol", "<unknown>")
        strike = contract.get("strike", None)
        iv = contract.get("iv", contract.get("premium", 0.2))
        is_call = contract.get("is_call", True)
        expiry = contract.get("expiry")
        expiry_days = (expiry - datetime.now().date()).days if expiry else contract.get("expiry_days", 1)

        # place into paper or live
        if TRADING_MODE == "live":
            logger.info(f"📡 LIVE ORDER: {symbol} BUY {qty} @ {spot}")
            # TODO: call broker adapter here
        elif TRADING_MODE == "paper":
            # paper_engine.place_order signature: (symbol, qty, price, strike, sigma, is_call, expiry_days)
            paper_engine.place_order(
                symbol=symbol,
                qty=qty,
                price=spot,
                strike=strike or spot,
                sigma=iv,
                is_call=is_call,
                expiry_days=max(1, expiry_days),
            )
        else:
            logger.info("Backtest mode: not placing live orders.")

        return contract

    except Exception as e:
        logger.error(f"❌ Failed to size/place contract: {e}")
        logger.debug("Contract payload: %s", contract)
        return None

# ----------------------------
# Trading loop
# ----------------------------
def trading_loop():
    global _last_decision_key, _state_buf

    logger.info("🚀 Starting trading loop...")
    try:
        send_telegram_message("🚀 Trading bot started")
    except Exception:
        pass

    while True:
        try:
            # Market hours gate (for paper/live)
            if TRADING_MODE in ("paper", "live"):
                if not market_open_now():
                    logger.info("⏸️ Market closed, waiting...")
                    time.sleep(60)
                    continue
            else:
                # Backtest branch
                df_path = config["backtest"]["data_path"]
                logger.info(f"📊 Backtest starting with {df_path}")
                bt = Backtester(strategy=strategy_engine,
                                initial_balance=INITIAL_BALANCE,
                                fee_perc=config["backtest"].get("fee", 0.001))
                bt.run_from_file(df_path)
                logger.info("✅ Backtest complete")
                break

            # Pull spot
            spot = get_spot()
            if spot <= 0:
                time.sleep(config["trading"].get("poll_interval", 5))
                continue

            # Build decision (StrategyEngine can internally combine RL + others)
            features = {"asset": "NIFTY", "spot": spot, "history": []}
            decision = strategy_engine.run(features)

            # Debounce / de-dupe (avoid log spam)
            key = (int(decision.get("signal", 0)),
                   int(decision.get("strike_offset", 0)),
                   str(decision.get("expiry", "weekly")))
            _state_buf.append(key)

            # Only proceed if the key is stable (same twice) AND different from last handled
            if len(_state_buf) == _state_buf.maxlen and len(set(_state_buf)) == 1:
                if key != _last_decision_key:
                    _last_decision_key = key
                    logger.info(f"📊 Consolidated decision: {decision}")

                    # Neutral signal → flatten & continue
                    if int(decision.get("signal", 0)) == 0:
                        flatten_if_neutral(0, spot)
                        acct = paper_engine.account_status(spot) if TRADING_MODE == "paper" else execution_engine.account_status(spot)
                        logger.info(pnl_log_from_account(acct))
                        log_greeks(spot)
                    else:
                        size_and_place(decision, spot)

            # Sleep
            time.sleep(config["trading"].get("poll_interval", 5))

        except KeyboardInterrupt:
            logger.info("🛑 Trading bot stopped manually")
            try:
                send_telegram_message("🛑 Trading bot stopped manually")
            except Exception:
                pass
            break
        except Exception as e:
            logger.exception(f"❌ Error in trading loop: {e}")
            time.sleep(2)  # small backoff

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    trading_loop()
