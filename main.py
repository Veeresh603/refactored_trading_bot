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

def size_and_place(decision: dict, spot: float):
    """
    Build contract, compute size with risk utilization logging,
    check risk, and place order via paper_engine/live adapter.
    """
    # 1) Build contract
    contract = options_optimizer.build_contract(spot, decision)
    logger.info(f"📦 Built contract: {contract}")

    # 2) Account & sizing
    acct = execution_engine.account_status(spot) if TRADING_MODE != "paper" else paper_engine.account_status(spot)
    equity = acct["total"]
    allowed_risk = equity * RISK_PER_TRADE

    premium = float(contract.get("premium", 0.0))
    if premium <= 0:
        # Fallback premium: approx using IV * 10 for a placeholder (rare)
        premium = max(1.0, float(contract.get("iv", 0.2)) * 100.0)

    lots = max(1, int(allowed_risk / (premium * LOT_SIZE)))
    side = "BUY" if int(decision["signal"]) > 0 else "SELL" if int(decision["signal"]) < 0 else "FLAT"

    notional_risk = lots * premium * LOT_SIZE
    utilization = 100.0 * notional_risk / allowed_risk if allowed_risk > 0 else 0.0

    logger.info(
        "⚖️ Position sizing: equity={:.2f}, allowed_risk={:.2f}, approx_premium={:.2f}, "
        "→ qty={} lots, side={} (utilization={:.1f}%)".format(
            equity, allowed_risk, premium, lots, side, utilization
        )
    )

    # 3) Risk checks (position caps, cool-off, daily loss, etc.)
    is_safe, reason = risk_manager.check_risk(
        equity=equity,
        pnl=acct["realized"] + acct["unrealized"],
        position_size=lots * LOT_SIZE
    )
    if not is_safe:
        logger.warning(f"⚠️ Risk triggered: {reason}")
        if risk_manager.can_hedge():
            logger.info("🛡️ Cool-off hedging allowed: skipping entry but will allow flatten/hedge trades.")
        else:
            risk_manager.start_cooloff()
            logger.warning(f"⏸️ Cool-off started for {risk_manager.cool_off_minutes} minutes")
        return

    if risk_manager.in_cooloff() and not risk_manager.can_hedge():
        logger.info("⏸️ Cool-off active, skipping this cycle")
        return

    # 4) Execute
    if side == "FLAT":
        flatten_if_neutral(0, spot)
    else:
        if TRADING_MODE == "live":
            # TODO: Add broker adapter call
            logger.info(f"📡 LIVE ORDER: {contract['symbol']} {side} {lots} @ {spot:.2f}")
            send_telegram_message(f"📡 LIVE ORDER: {contract['symbol']} {side} {lots} @ {spot:.2f}")
        else:
            if side == "BUY":
                paper_engine.buy(contract, lots, spot)
            else:
                paper_engine.sell(contract, lots, spot)

    # 5) Log PnL & Greeks
    acct2 = execution_engine.account_status(spot) if TRADING_MODE != "paper" else paper_engine.account_status(spot)
    logger.info(pnl_log_from_account(acct2))
    log_greeks(spot)

    # 6) Optional: pretty-print open positions when small
    try:
        paper_engine.log_positions()
    except Exception:
        pass

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
