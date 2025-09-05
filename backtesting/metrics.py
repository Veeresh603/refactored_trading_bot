"""
Backtesting Metrics (robust)
----------------------------
- Accepts a number of inputs:
  * pd.DataFrame with columns ["time", "equity"]
  * pd.Series (index=time or integer) -> treated as equity series
  * list/np.array of equity values -> synthetic time index generated
  * TradingEnv-like object exposing:
      - equity_curve (pd.Series / pd.DataFrame) OR
      - get_equity_curve() -> pd.Series / pd.DataFrame OR
      - reward_history / pnl_history -> will be cumsum() -> equity
- Outputs a dict with common performance metrics (or {} if not enough data)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional


def _to_equity_df(obj: Any) -> Optional[pd.DataFrame]:
    """
    Normalize various inputs into a pd.DataFrame with columns ["time","equity"].
    Returns None if conversion not possible or insufficient data.
    """
    # If already a DataFrame with 'equity'
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        # If single-column Series-like DataFrame, allow it as equity
        if "equity" in df.columns:
            if "time" not in df.columns:
                # Use index as time if possible
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index().rename(columns={df.index.name or "index": "time"})
                else:
                    # create synthetic times
                    df = df.reset_index(drop=True)
                    df["time"] = pd.date_range(start=pd.Timestamp.now(), periods=len(df), freq="D")
                df = df[["time", "equity"]]
            else:
                df = df[["time", "equity"]]
            return df

        # Try common alternatives: columns named 'value' or 0th column
        for col in ["value", "equity_curve", "balance", "portfolio_value"]:
            if col in df.columns:
                eq = df[[col]].copy()
                eq.columns = ["equity"]
                if "time" in df.columns:
                    eq["time"] = df["time"].values
                else:
                    if isinstance(df.index, pd.DatetimeIndex):
                        eq["time"] = df.index
                    else:
                        eq["time"] = pd.date_range(start=pd.Timestamp.now(), periods=len(eq), freq="D")
                return eq[["time", "equity"]]

        # If DataFrame is single-column with numeric index, treat that column as equity
        if df.shape[1] == 1:
            col = df.columns[0]
            eq = df.reset_index(drop=True).rename(columns={col: "equity"})
            eq["time"] = pd.date_range(start=pd.Timestamp.now(), periods=len(eq), freq="D")
            return eq[["time", "equity"]]

        return None

    # If it's a Series -> make DataFrame
    if isinstance(obj, pd.Series):
        s = obj.copy()
        if s.empty:
            return None
        if isinstance(s.index, pd.DatetimeIndex):
            df = s.reset_index()
            df.columns = ["time", "equity"]
            return df[["time", "equity"]]
        else:
            df = pd.DataFrame({"equity": s.values})
            df["time"] = pd.date_range(start=pd.Timestamp.now(), periods=len(df), freq="D")
            return df[["time", "equity"]]

    # If it's array-like (list/np.array)
    if isinstance(obj, (list, tuple, np.ndarray)):
        arr = np.asarray(obj, dtype=float)
        if arr.size == 0:
            return None
        df = pd.DataFrame({"equity": arr})
        df["time"] = pd.date_range(start=pd.Timestamp.now(), periods=len(arr), freq="D")
        return df[["time", "equity"]]

    # If it's an env-like object, try to extract common attributes
    # Prefer: equity_curve (Series/DataFrame) -> get_equity_curve()
    # Then: reward_history / pnl_history -> cumsum()
    try:
        # 1) direct attribute
        for attr in ("equity_curve", "equity", "account_equity", "equity_history"):
            if hasattr(obj, attr):
                candidate = getattr(obj, attr)
                res = _to_equity_df(candidate)
                if res is not None:
                    return res

        # 2) getter method
        for fn in ("get_equity_curve", "get_equity", "get_account_equity", "get_portfolio_value"):
            if hasattr(obj, fn) and callable(getattr(obj, fn)):
                try:
                    candidate = getattr(obj, fn)()
                    res = _to_equity_df(candidate)
                    if res is not None:
                        return res
                except Exception:
                    # swallow and continue
                    pass

        # 3) reward/pnl history -> cumsum -> equity
        for hist in ("reward_history", "rewards", "returns", "pnl_history", "upl_history", "profit_history"):
            if hasattr(obj, hist):
                candidate = getattr(obj, hist)
                try:
                    series = pd.Series(candidate).astype(float).cumsum()
                    return _to_equity_df(series)
                except Exception:
                    pass

        # 4) trades list -> try to extract 'equity' or running 'balance'
        if hasattr(obj, "trades"):
            trades = getattr(obj, "trades")
            if isinstance(trades, (list, tuple)) and trades:
                # look for 'equity' or 'balance' or cumulative 'pnl'
                if isinstance(trades[0], dict):
                    if "equity" in trades[0]:
                        try:
                            series = pd.Series([t["equity"] for t in trades]).astype(float)
                            return _to_equity_df(series)
                        except Exception:
                            pass
                    if "balance" in trades[0]:
                        try:
                            series = pd.Series([t["balance"] for t in trades]).astype(float)
                            return _to_equity_df(series)
                        except Exception:
                            pass
                    if "pnl" in trades[0] or "profit" in trades[0]:
                        try:
                            series = pd.Series([t.get("pnl", t.get("profit", 0.0)) for t in trades]).astype(float).cumsum()
                            return _to_equity_df(series)
                        except Exception:
                            pass

    except Exception:
        # any extraction attempt should not crash metrics
        pass

    return None


def compute_metrics(equity_curve: Any, risk_free_rate: float = 0.05) -> Dict[str, float]:
    """
    Compute performance metrics from equity curve.

    Accepts:
      - pd.DataFrame with ["time","equity"]
      - pd.Series (index=time or numeric)
      - list/np.array of equity values
      - env-like object (TradingEnv) exposing equity or pnl history

    Returns:
        dict: performance metrics (empty dict when insufficient data)
    """
    df = _to_equity_df(equity_curve)
    if df is None or df.empty:
        return {}

    # Ensure proper dtypes
    df = df.copy()
    # try parsing time to datetime if not already
    if not np.issubdtype(df["time"].dtype, np.datetime64):
        try:
            df["time"] = pd.to_datetime(df["time"])
        except Exception:
            # fallback: create synthetic monotonic dates
            df["time"] = pd.date_range(start=pd.Timestamp.now(), periods=len(df), freq="D")

    df = df.sort_values("time").reset_index(drop=True)

    if "equity" not in df.columns or len(df) < 2:
        return {}

    equity = df["equity"].astype(float).values

    # If any non-finite values, drop them
    mask = np.isfinite(equity)
    if np.sum(mask) < 2:
        return {}
    equity = equity[mask]
    # recompute returns
    returns = np.diff(equity) / (equity[:-1] + 1e-12)

    metrics: Dict[str, float] = {}
    # Safe guards for division by zero and short series
    try:
        metrics["total_return"] = float(equity[-1] / (equity[0] + 1e-12) - 1.0)
    except Exception:
        metrics["total_return"] = 0.0

    # CAGR calculation
    try:
        days = (df["time"].iloc[-1] - df["time"].iloc[0]).days
        if days <= 0:
            metrics["cagr"] = 0.0
        else:
            metrics["cagr"] = float((equity[-1] / (equity[0] + 1e-12)) ** (252.0 / days) - 1.0)
    except Exception:
        metrics["cagr"] = 0.0

    # Volatility (annualized)
    if returns.size > 0:
        vol = float(np.std(returns) * np.sqrt(252.0))
    else:
        vol = 0.0
    metrics["volatility"] = vol

    # Sharpe & Sortino
    def _calc_sharpe_local(r, rf):
        if r.size == 0:
            return 0.0
        excess = r - (rf / 252.0)
        denom = np.std(excess) + 1e-9
        return float(np.mean(excess) / denom * np.sqrt(252.0))

    def _calc_sortino_local(r, rf):
        if r.size == 0:
            return 0.0
        downside = r[r < 0.0]
        if downside.size == 0:
            return float("inf")
        excess = r - (rf / 252.0)
        denom = np.std(downside) + 1e-9
        return float(np.mean(excess) / denom * np.sqrt(252.0))

    metrics["sharpe"] = _calc_sharpe_local(returns, risk_free_rate)
    metrics["sortino"] = _calc_sortino_local(returns, risk_free_rate)

    # Win rate, avg win/loss, profit factor
    if returns.size > 0:
        wins = returns[returns > 0.0]
        losses = returns[returns < 0.0]
        metrics["win_rate"] = float(wins.size / returns.size)
        metrics["avg_win"] = float(np.mean(wins)) if wins.size > 0 else 0.0
        metrics["avg_loss"] = float(np.mean(losses)) if losses.size > 0 else 0.0
        metrics["profit_factor"] = float(
            (np.sum(wins) / (-np.sum(losses) + 1e-12)) if np.sum(losses) != 0 else np.inf
        )
    else:
        metrics["win_rate"] = 0.0
        metrics["avg_win"] = 0.0
        metrics["avg_loss"] = 0.0
        metrics["profit_factor"] = 0.0

    # Max drawdown
    def _calc_max_drawdown_local(e):
        peak = e[0]
        max_dd = 0.0
        for val in e:
            if val > peak:
                peak = val
            dd = (peak - val) / (peak + 1e-12)
            if dd > max_dd:
                max_dd = dd
        return float(max_dd)

    metrics["max_drawdown"] = _calc_max_drawdown_local(equity)

    return metrics
