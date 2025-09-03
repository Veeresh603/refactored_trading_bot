"""
Utils
-----
General-purpose utilities:
- Telegram messaging
- File sending
- Logging helpers
- Common time/dir helpers
"""

import os
import logging
import requests
from datetime import datetime

logger = logging.getLogger("Utils")

# ----------------------------
# Telegram Helpers
# ----------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram_message(msg: str):
    """Send text message to Telegram (if enabled)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug(f"Telegram not configured. Message skipped: {msg}")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
        if resp.status_code != 200:
            logger.warning(f"Telegram message failed: {resp.text}")
    except Exception as e:
        logger.error(f"Telegram send error: {e}")


def send_telegram_file(filepath: str, caption: str = ""):
    """Send file (PDF/CSV/IMG) to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured. File skipped.")
        return
    if not os.path.exists(filepath):
        logger.error(f"File not found for Telegram send: {filepath}")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
        with open(filepath, "rb") as f:
            resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption}, files={"document": f})
        if resp.status_code != 200:
            logger.warning(f"Telegram file send failed: {resp.text}")
    except Exception as e:
        logger.error(f"Telegram file send error: {e}")


# ----------------------------
# Generic Helpers
# ----------------------------
def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)
    return path


def timestamp():
    """Return current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_float(val, default=0.0):
    """Convert to float safely."""
    try:
        return float(val)
    except Exception:
        return default
