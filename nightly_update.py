import os
import datetime as dt
import subprocess
import logging
from dotenv import load_dotenv

from core.data_downloader import (
    load_config,
    download_zerodha,
    download_angelone,
    download_nse_option_chain,
    update_csv,
)
from core.utils import send_telegram_message

# -------------------------
# Setup
# -------------------------
os.makedirs("logs", exist_ok=True)
load_dotenv()

logging.basicConfig(
    filename="logs/nightly_update.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("NightlyUpdate")


# -------------------------
# Step 1: Update Historical Data
# -------------------------
def update_data():
    config = load_config()
    broker = config.get("data", {}).get("source", "nse").lower()
    output_file = config.get("data", {}).get("output_file", "data/nifty.csv")

    today = dt.date.today()
    from_date = today - dt.timedelta(days=5)
    to_date = today

    if broker == "zerodha":
        instrument_token = config["data"].get("instrument_token", 256265)
        new_data = download_zerodha(instrument_token, from_date, to_date, interval="day")
        update_csv(output_file, new_data, date_col="date")

    elif broker == "angelone":
        symboltoken = config["data"].get("symboltoken", "3045")
        from_dt = dt.datetime.combine(from_date, dt.time(9, 15))
        to_dt = dt.datetime.combine(to_date, dt.time(15, 30))
        new_data = download_angelone(symboltoken, from_dt, to_dt)
        update_csv(output_file, new_data, date_col="datetime")

    elif broker == "nse":
        symbol = config["data"].get("symbol", "NIFTY")
        new_data = download_nse_option_chain(symbol)
        update_csv(output_file, new_data, date_col="expiry")

    else:
        raise ValueError(f"❌ Unknown data source: {broker}")

    logger.info(f"Data update successful for {broker}, saved to {output_file}")


# -------------------------
# Step 2: Rotate Checkpoints
# -------------------------
def rotate_checkpoints():
    os.makedirs("models/archive", exist_ok=True)
    model_path = "models/best_allocator_strike_expiry.zip"
    if os.path.exists(model_path):
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
        archived_path = f"models/archive/allocator_{ts}.zip"
        os.rename(model_path, archived_path)
        logger.info(f"Archived old allocator checkpoint -> {archived_path}")


# -------------------------
# Step 3: Retrain RL Allocator
# -------------------------
def retrain_allocator():
    logger.info("Starting RL Allocator retraining...")
    subprocess.run(["python", "strategies/train_allocator.py"], check=True)
    logger.info("RL Allocator retraining completed.")


# -------------------------
# Step 4: Refresh Risk Parameters
# -------------------------
def refresh_risk_params():
    config = load_config()
    risk = config.get("risk", {})
    risk_file = "logs/risk_params.log"

    with open(risk_file, "w") as f:
        for key, value in risk.items():
            f.write(f"{key}: {value}\n")

    logger.info(f"Risk parameters refreshed from config.yaml -> {risk}")
    return risk


# -------------------------
# Main Entry
# -------------------------
if __name__ == "__main__":
    try:
        logger.info("🚀 Nightly update started")

        update_data()
        rotate_checkpoints()
        retrain_allocator()
        risk = refresh_risk_params()

        msg = f"✅ Nightly update completed.\nRisk Params: {risk}"
        logger.info(msg)
        send_telegram_message(msg)

    except Exception as e:
        msg = f"❌ Nightly update failed: {e}"
        logger.error(msg)
        send_telegram_message(msg)
