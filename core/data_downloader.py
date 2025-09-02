import os
import pandas as pd
import datetime as dt
import yaml
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Config Loader
# -------------------------
def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------
# Zerodha Downloader
# -------------------------
def download_zerodha(instrument_token, from_date, to_date, interval="day"):
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        raise ImportError("kiteconnect not installed. Run: pip install kiteconnect")

    kite = KiteConnect(api_key=os.getenv("KITE_API_KEY"))
    kite.set_access_token(os.getenv("KITE_ACCESS_TOKEN"))

    data = kite.historical_data(instrument_token, from_date, to_date, interval)
    return pd.DataFrame(data)


# -------------------------
# Angel One Downloader
# -------------------------
def download_angelone(symboltoken, from_date, to_date, interval="FIFTEEN_MINUTE"):
    try:
        from SmartApi import SmartConnect
    except ImportError:
        raise ImportError("smartapi-python not installed. Run: pip install smartapi-python")

    api_key = os.getenv("ANGEL_API_KEY")
    client = SmartConnect(api_key=api_key)
    client.generateSession(os.getenv("ANGEL_CLIENT_ID"), os.getenv("ANGEL_PASSWORD"), os.getenv("ANGEL_TOTP"))

    params = {
        "exchange": "NSE",
        "symboltoken": symboltoken,
        "interval": interval,
        "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
        "todate": to_date.strftime("%Y-%m-%d %H:%M")
    }

    data = client.getCandleData(params)
    return pd.DataFrame(data["data"])


# -------------------------
# NSE Free Option Chain Snapshot
# -------------------------
def download_nse_option_chain(symbol="NIFTY"):
    import requests

    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers)
    data = session.get(url, headers=headers).json()

    records = []
    for opt in data["records"]["data"]:
        if "CE" in opt:
            ce = opt["CE"]
            records.append([
                opt["expiryDate"],
                ce["strikePrice"],
                ce["lastPrice"],
                ce["impliedVolatility"],
                ce["openInterest"]
            ])

    return pd.DataFrame(records, columns=["expiry", "strike", "close", "iv", "oi"])


# -------------------------
# CSV Updater
# -------------------------
def update_csv(output_file, new_data, date_col="date"):
    """Append new rows to CSV, avoiding duplicates"""
    if os.path.exists(output_file):
        old_df = pd.read_csv(output_file, parse_dates=[date_col])
        combined = pd.concat([old_df, new_data])
        combined = combined.drop_duplicates(subset=[date_col])
        combined.to_csv(output_file, index=False)
        print(f"✅ Updated {output_file} with {len(new_data)} new rows")
    else:
        new_data.to_csv(output_file, index=False)
        print(f"✅ Created new file {output_file}")


# -------------------------
# Main Entry
# -------------------------
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    config = load_config()

    broker = config.get("data", {}).get("source", "nse").lower()
    output_file = config.get("data", {}).get("output_file", "data/nifty.csv")

    today = dt.date.today()
    from_date = today - dt.timedelta(days=5)
    to_date = today

    if broker == "zerodha":
        instrument_token = config["data"].get("instrument_token", 256265)  # default: NIFTY
        new_data = download_zerodha(instrument_token, from_date, to_date, interval="day")
        update_csv(output_file, new_data, date_col="date")

    elif broker == "angelone":
        symboltoken = config["data"].get("symboltoken", "3045")  # default: NIFTY
        new_data = download_angelone(symboltoken, dt.datetime.combine(from_date, dt.time(9, 15)), dt.datetime.combine(to_date, dt.time(15, 30)))
        update_csv(output_file, new_data, date_col="datetime")

    elif broker == "nse":
        symbol = config["data"].get("symbol", "NIFTY")
        new_data = download_nse_option_chain(symbol)
        update_csv(output_file, new_data, date_col="expiry")

    else:
        raise ValueError(f"❌ Unknown data source: {broker}")
