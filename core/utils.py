"""
Utils Module
------------
- Logging
- Config loader
- Telegram alerts
"""

import logging
import yaml
import requests
import os


# -------------------------------
# Logging Setup
# -------------------------------
def setup_logger(name="TradingBot", log_file="logs/trading_bot.log", level=logging.INFO):
    """
    Setup logger with file + console output
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# -------------------------------
# Config Loader
# -------------------------------
def load_config(path="config/config.yaml"):
    """
    Load config from YAML file
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------------
# Telegram Alerts
# -------------------------------
def send_telegram_message(message, bot_token=None, chat_id=None):
    """
    Send a Telegram alert
    """
    bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        print("⚠️ Telegram credentials not found in ENV.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}

    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print(f"📩 Telegram message sent: {message}")
            return True
        else:
            print(f"❌ Telegram send failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False
