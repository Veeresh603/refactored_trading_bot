"""
Nightly Update Script
---------------------
- Downloads fresh historical data (Zerodha, AngelOne, NSE Option Chain)
- Updates CSV files
- Logs activity and sends Telegram alerts
"""

import os
import datetime as dt
import logging
import time
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
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("NightlyUpdate")


# -------------------------
# Helper
# -------------------------
def safe_run(task_name, func, *args, retries=3, delay=5, **kwargs):
    """
    Run a task safely with retries.
    """
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"▶️ Running {task_name} (attempt {attempt})")
            result = func(*args, **kwargs)
            logger.info(f"✅ {task_name} completed")
            return result
        except Exception as e:
            logger.error(f"❌ {task_name} failed (attempt {attempt}): {e}")
            time.sleep(delay)

    send_telegram_message(f"❌ Nightly Update: {task_name} failed after {retries} retries")
    return None


# -------------------------
# Main Update Process
# -------------------------
def main():
    logger.info("🚀 Starting nightly update")
    send_telegram_message("🌙 Nightly update started")

    config = load_config()

    # Zerodha
    if config.get("data", {}).get("zerodha", True):
        safe_run("Zerodha Data Download", download_zerodha, config)

    # AngelOne
    if config.get("data", {}).get("angelone", False):
        safe_run("AngelOne Data Download", download_angelone, config)

    # NSE Option Chain
    if config.get("data", {}).get("nse_option_chain", True):
        safe_run("NSE Option Chain Download", download_nse_option_chain, config)

    # Update CSVs
    safe_run("CSV Update", update_csv, config)

    logger.info("🌙 Nightly update completed")
    send_telegram_message("✅ Nightly update completed successfully")


# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Nightly update crashed: {e}")
        send_telegram_message(f"❌ Nightly update crashed: {e}")
