"""
Generate Reports
----------------
- End-of-day trade reports
- Export to CSV + summary
"""

import pandas as pd
import os
from datetime import datetime


def generate_trade_report(trades: pd.DataFrame, output_dir="reports/output"):
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Save full trades to CSV
    csv_path = os.path.join(output_dir, f"trades_{date_str}.csv")
    trades.to_csv(csv_path, index=False)

    # Summary
    total_pnl = trades["pnl"].sum()
    win_rate = (trades["pnl"] > 0).mean() * 100
    avg_pnl = trades["pnl"].mean()

    summary = {
        "Date": date_str,
        "Total Trades": len(trades),
        "Total PnL": total_pnl,
        "Win Rate (%)": win_rate,
        "Avg PnL": avg_pnl
    }

    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, f"summary_{date_str}.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"✅ Trade report generated: {csv_path}")
    print(f"✅ Summary report generated: {summary_path}")
    return summary_df
# Reporting script placeholder
