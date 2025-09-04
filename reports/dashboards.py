import streamlit as st
import pandas as pd
import time
from core import execution_engine

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Trading Bot Dashboard", page_icon="📊", layout="wide")
st.title("📊 Live Trading Dashboard")

placeholder = st.empty()

# Dummy allocator weights (replace with live RL allocator later)
allocator_weights = {
    ("NIFTY", "SMA"): 0.3,
    ("NIFTY", "RSI"): 0.4,
    ("NIFTY", "RL"): 0.3,
}

# ----------------------------
# Dashboard loop
# ----------------------------
while True:
    with placeholder.container():
        spot_price = 20000.0  # replace with live WebSocket

        # Account status
        status = execution_engine.account_status(spot_price)

        # Greeks
        delta, gamma, vega, theta = execution_engine.portfolio_greeks(spot_price)

        # Get live trade history from execution_engine
        trade_df = execution_engine.get_trade_history(limit=10)

        # ----------------------------
        # Metrics Row
        # ----------------------------
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("💰 Balance", f"{status['balance']:.2f}")
        col2.metric("📌 Margin Used", f"{status['margin_used']:.2f}")
        col3.metric("PnL Realized", f"{status['realized']:.2f}")
        col4.metric("PnL Unrealized", f"{status['unrealized']:.2f}")
        col5.metric("PnL Total", f"{status['total']:.2f}")

        # ----------------------------
        # Greeks
        # ----------------------------
        st.subheader("📈 Portfolio Greeks")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Δ Delta", f"{delta:.2f}")
        g2.metric("Γ Gamma", f"{gamma:.4f}")
        g3.metric("V Vega", f"{vega:.2f}")
        g4.metric("Θ Theta", f"{theta:.2f}")

        # ----------------------------
        # Allocator Weights
        # ----------------------------
        st.subheader("⚖️ Allocator Weights")
        weight_df = pd.DataFrame.from_dict(allocator_weights, orient="index", columns=["Weight"])
        st.bar_chart(weight_df)

        # ----------------------------
        # Trade History
        # ----------------------------
        st.subheader("📜 Trade History (last 10)")
        st.table(trade_df)

    time.sleep(5)
