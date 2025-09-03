"""
Analyze Results + Report Generator + Telegram Sending
-----------------------------------------------------
- Loads backtest/live results
- Plots equity curve & drawdowns
- Computes performance metrics
- Generates PDF/HTML reports
- Sends report to Telegram
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from backtesting.metrics import compute_metrics
from fpdf import FPDF
import requests
from core.utils import send_telegram_message


def send_telegram_file(file_path, caption="Trading Report"):
    """Send a file (PDF/HTML) to Telegram."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("⚠️ Telegram not configured")
        return False

    url = f"https://api.telegram.org/bot{token}/sendDocument"
    with open(file_path, "rb") as f:
        files = {"document": f}
        data = {"chat_id": chat_id, "caption": caption}
        resp = requests.post(url, data=data, files=files)
        if resp.status_code == 200:
            print(f"✅ Report sent to Telegram: {file_path}")
            return True
        else:
            print(f"❌ Telegram file send failed: {resp.text}")
            return False


def analyze_results(results_path="results/equity_curve.csv", trades_path="results/trades.csv", generate_report=True):
    if not os.path.exists(results_path) or not os.path.exists(trades_path):
        print("❌ Results not found. Run backtest or trading first.")
        return

    equity_df = pd.read_csv(results_path, parse_dates=["time"])
    trades_df = pd.read_csv(trades_path)

    if equity_df.empty:
        print("❌ Equity curve is empty")
        return

    # Compute metrics
    metrics = compute_metrics(equity_df.rename(columns={"equity": "equity"}))

    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df["time"], equity_df["equity"], label="Equity Curve", color="blue")
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid()
    equity_plot = "results/equity_curve.png"
    plt.savefig(equity_plot)
    plt.close()

    # Plot drawdowns
    peak = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] - peak) / peak
    plt.figure(figsize=(12, 4))
    plt.fill_between(equity_df["time"], drawdown, color="red", alpha=0.4)
    plt.title("Drawdowns")
    plt.ylabel("Drawdown")
    plt.grid()
    dd_plot = "results/drawdowns.png"
    plt.savefig(dd_plot)
    plt.close()

    # Print metrics
    print("\n📊 Performance Metrics")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.4f}" if isinstance(v, (int, float)) else f"{k:20s}: {v}")

    # Trade summary
    if not trades_df.empty:
        print("\n📈 Trade Summary")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Wins: {(trades_df['pnl'] > 0).sum()} | Losses: {(trades_df['pnl'] <= 0).sum()}")
    else:
        print("No trades found.")

    # -----------------------------
    # PDF Report Generation
    # -----------------------------
    pdf_path = None
    if generate_report:
        os.makedirs("results/reports", exist_ok=True)
        pdf_path = "results/reports/trading_report.pdf"

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Trading Performance Report", ln=True, align="C")

        # Equity Curve Image
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Equity Curve", ln=True)
        pdf.image(equity_plot, x=10, y=None, w=180)

        # Drawdown Image
        pdf.cell(0, 10, "Drawdowns", ln=True)
        pdf.image(dd_plot, x=10, y=None, w=180)

        # Metrics
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Performance Metrics", ln=True)
        pdf.set_font("Arial", "", 10)
        for k, v in metrics.items():
            pdf.cell(0, 8, f"{k}: {v}", ln=True)

        # Trade Summary
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Trade Summary", ln=True)
        pdf.set_font("Arial", "", 10)
        if not trades_df.empty:
            pdf.cell(0, 8, f"Total Trades: {len(trades_df)}", ln=True)
            pdf.cell(0, 8, f"Wins: {(trades_df['pnl'] > 0).sum()} | Losses: {(trades_df['pnl'] <= 0).sum()}", ln=True)
        else:
            pdf.cell(0, 8, "No trades found", ln=True)

        pdf.output(pdf_path)
        print(f"\n📄 Report generated: {pdf_path}")

    # -----------------------------
    # Send to Telegram
    # -----------------------------
    if pdf_path:
        caption = f"📊 Trading Report\nTotal Return: {metrics.get('total_return', 0):.2%}\nSharpe: {metrics.get('sharpe', 0):.2f}"
        send_telegram_message("📩 Trading report ready. Uploading PDF...")
        send_telegram_file(pdf_path, caption=caption)


def main():
    analyze_results()


if __name__ == "__main__":
    main()
