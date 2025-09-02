import fastbt
import pandas as pd

def get_strategy_returns(strategy, df, initial_balance=100000):
    """
    Run a backtest for a given strategy and return daily returns
    """
    strat_test = strategy.generate_signals(df.copy())

    signal_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
    signals = [signal_map.get(s, 0) for s in strat_test["signal"].tolist()]

    trades = fastbt.backtest(strat_test["close"].tolist(), signals)

    equity_curve = [initial_balance]
    for t in trades:
        equity_curve.append(equity_curve[-1] + t.pnl)

    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().fillna(0)
    return returns
