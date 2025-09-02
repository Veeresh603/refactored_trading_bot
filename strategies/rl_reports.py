import os
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF


def generate_report(metrics, balances, rewards, report_path="reports/rl_report.pdf"):
    """
    Generate PDF report with trading metrics and plots
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # --- Equity Curve ---
    plt.figure(figsize=(8, 4))
    pd.Series(balances).plot(title="Equity Curve", grid=True)
    plt.ylabel("Balance")
    plt.tight_layout()
    equity_curve_img = "reports/equity_curve.png"
    plt.savefig(equity_curve_img)
    plt.close()

    # --- Reward Distribution ---
    plt.figure(figsize=(6, 4))
    pd.Series(rewards).hist(bins=30, grid=False, alpha=0.7)
    plt.title("Trade Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.tight_layout()
    rewards_img = "reports/rewards_hist.png"
    plt.savefig(rewards_img)
    plt.close()

    # --- PDF Report ---
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "RL Walk-Forward Training Report", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Sharpe Ratio: {metrics['sharpe']:.2f}", ln=True)
    pdf.cell(200, 10, f"Win Rate: {metrics['win_rate']*100:.2f}%", ln=True)
    pdf.cell(200, 10, f"Max Drawdown: {metrics['max_dd']:.2%}", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, "Equity Curve:", ln=True)
    pdf.image(equity_curve_img, x=10, w=180)

    pdf.ln(5)
    pdf.cell(200, 10, "Reward Distribution:", ln=True)
    pdf.image(rewards_img, x=30, w=150)

    pdf.output(report_path)
    print(f"✅ RL training report generated: {report_path}")
