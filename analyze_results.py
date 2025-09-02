import pandas as pd
import matplotlib.pyplot as plt

# Load results
results = pd.read_csv("backtest_results.csv", parse_dates=["date"])
summary = pd.read_csv("backtest_summary.csv").iloc[0]

# Plot equity curve
plt.figure(figsize=(12, 6))
plt.plot(results["date"], results["equity"], label="Equity Curve", color="blue")
plt.title("RL Allocator Backtest Equity Curve")
plt.xlabel("Date")
plt.ylabel("Equity (INR)")
plt.legend()
plt.grid()
plt.show()

# Print metrics
print("\n📊 Performance Metrics")
print(f"Total Return: {summary['total_return']:.2%}")
print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {summary['max_drawdown']:.2%}")
print(f"Win Rate: {summary['win_rate']:.2%}")
