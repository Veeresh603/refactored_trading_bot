from ai import train_rl_allocator
from core.data_downloader import DataDownloader
from core.broker import Broker
import pandas as pd


def nightly_update():
    print("🌙 Nightly Allocator Update Starting...")

    broker = Broker()
    kite = broker.connect()
    downloader = DataDownloader(broker)

    instruments = {
        "NIFTY": 738561,
        "BANKNIFTY": 260105,
    }

    all_data = {}
    greek_series = {"delta": [], "gamma": [], "vega": []}

    for asset, token in instruments.items():
        print(f"⬇️ Downloading {asset} OHLCV...")
        df = downloader.download_ohlcv(token, days=180, interval="1day")
        if df.empty:
            continue

        # Save OHLCV
        path = f"data/{asset.lower()}_latest.csv"
        df.to_csv(path)
        all_data[asset] = df

        # Build Greeks from option chain snapshots (placeholder: live fetch needed)
        deltas, gammas, vegas = [], [], []
        for idx, row in df.iterrows():
            spot = row["close"]

            # ❗ Placeholder: fetch actual option chain snapshot here
            option_chain = []  # TODO: fetch from NSE/Zerodha

            d, g, v = downloader.compute_greeks_from_chain(option_chain, spot)
            deltas.append(d)
            gammas.append(g)
            vegas.append(v)

        greek_series["delta"] = pd.Series(deltas, index=df.index)
        greek_series["gamma"] = pd.Series(gammas, index=df.index)
        greek_series["vega"] = pd.Series(vegas, index=df.index)

    # Returns
    train_returns = {}
    for asset, df in all_data.items():
        train_returns[(asset, "SMA")] = df["close"].pct_change().fillna(0)
        train_returns[(asset, "RSI")] = df["close"].pct_change().fillna(0) * 0.8
        train_returns[(asset, "RL")] = df["close"].pct_change().fillna(0) * 1.2

    # Step 3: Train Allocator
    print("📈 Training Allocator...")
    train_rl_allocator(train_returns, greek_series)

    # Step 4: Evaluate checkpoints
    print("🔍 Evaluating Checkpoints...")
    evaluate_checkpoints(train_returns, greek_series)

    print("✅ Nightly Update Complete!")
    send_telegram_message("🌙 Nightly Allocator Update Complete — Best Model Ready!")
