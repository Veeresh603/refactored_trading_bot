"""
RL Reports
----------
- Generates PDF/HTML reports for RL training
- Uses shared metrics (from backtesting/metrics)
- Avoids duplication with analyze_results
- Optional Telegram sending
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
from backtesting.metrics import compute_metrics
from core.utils import send_telegram_message
from analyze_results import send_telegram_file


def generate_rl_report(metrics, balances, rewards, report_path="results/reports/rl_report.pdf", send_telegram=False):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # --- Equity Curve ---
    plt.figure(figsize=(8, 4))
    pd.Series(balances).plot(title="Equity Curve", grid=True)
    plt.ylabel("Balance")
    plt.tight_layout()
    equity_curve_img = "results/reports/rl_equity_curve.png"
    plt.savefig(equity_curve_img)
    plt.close()

    # --- Rewards ---
    plt.figure(figsize=(8, 4))
    pd.Series(rewards).plot(title="Episode Rewards", grid=True)
    plt.ylabel("Reward")
    plt.tight_layout()
    reward_img = "results/reports/rl_rewards.png"
    plt.savefig(reward_img)
    plt.close()

    # --- PDF Report ---
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "RL Training Report", ln=True, align="C")

    # Metrics
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Performance Metrics", ln=True)
    pdf.set_font("Arial", "", 10)
    for k, v in metrics.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)

    # Plots
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Equity Curve", ln=True)
    pdf.image(equity_curve_img, x=10, y=None, w=180)
    pdf.cell(0, 10, "Rewards", ln=True)
    pdf.image(reward_img, x=10, y=None, w=180)

    pdf.output(report_path)
    print(f"📄 RL report saved: {report_path}")

    # --- Telegram ---
    if send_telegram:
        caption = f"🤖 RL Training Report\nFinal Balance={balances[-1]:.2f}\nMean Reward={pd.Series(rewards).mean():.2f}"
        send_telegram_message("📩 Uploading RL training report...")
        send_telegram_file(report_path, caption=caption)
