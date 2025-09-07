# backtesting/backtest.py
"""
Backtester with partial-fill semantics and optional orderbook sampler + TCA integration.

Key fixes in this version:
- When volume is present, available liquidity = bar_volume * liquidity_fraction (no forced min)
  -> enables partial fills when liquidity is low (tests expect this).
- When volume is missing or zero, fall back to order_unit_size so fills can still occur.
- Keeps prior fixes: metrics keys, allow_short flag, order queue semantics, fill delay, slippage, commission.
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple
import logging
import math

import numpy as np
import pandas as pd

# Placeholders for optional modules (if present in your project)
try:
    from backtesting.orderbook_sampler import OrderBookSampler
except Exception:
    OrderBookSampler = None

try:
    from backtesting.tca import estimate_impact_from_params
except Exception:
    def estimate_impact_from_params(*_, **__):
        return 0.0

logger = logging.getLogger("TradingBot.Backtester")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def _max_drawdown_from_series(eq: np.ndarray) -> float:
    """Compute maximum drawdown from numpy equity series (as fraction)."""
    if len(eq) == 0:
        return 0.0
    peak = -np.inf
    max_dd = 0.0
    for v in eq:
        if v > peak:
            peak = v
        dd = (peak - v) / (peak + 1e-12)
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def _sharpe_from_returns(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    mu = float(np.mean(returns))
    sigma = float(np.std(returns))
    if sigma <= 0.0:
        return 0.0
    # annualize assuming 252 trading days
    return float(mu / sigma * math.sqrt(252.0))


class Backtester:
    def __init__(
        self,
        strategy: Any,
        initial_balance: float = 100000.0,
        fill_delay_steps: int = 1,
        slippage_pct: float = 0.0,
        commission: float = 0.0,
        position_size: float = 1.0,
        liquidity_fraction: float = 0.1,
        orderbook_sampler: Optional["OrderBookSampler"] = None,
        tca_params: Optional[Dict[str, float]] = None,
        allow_short: bool = False,
    ):
        self.strategy = strategy
        self.initial_balance = float(initial_balance)
        self.fill_delay_steps = int(fill_delay_steps)
        if self.fill_delay_steps < 0:
            raise ValueError("fill_delay_steps must be >= 0")
        self.slippage_pct = float(slippage_pct)
        self.commission = float(commission)
        self.position_size = float(position_size)
        self.liquidity_fraction = float(liquidity_fraction)
        if not (0.0 <= self.liquidity_fraction <= 1.0):
            raise ValueError("liquidity_fraction must be in [0,1]")
        self.orderbook_sampler = orderbook_sampler
        self.tca_params = tca_params or {"a": 0.0, "b": 0.0}
        self.allow_short = bool(allow_short)

    def _normalize_signals(self, sig: Iterable[int]) -> np.ndarray:
        """
        Normalize various signal return types into an integer numpy array with values in {-1,0,1}.
        Accepts:
          - python iterables (lists, tuples, generators) of ints
          - numpy arrays
          - pandas.Series (will be converted to ints)
          - pandas.DataFrame (tries common column names 'signal','signals','position','pos','action',
            or uses the first column if the DF has exactly one column).
        After extraction, clamps values >1 -> 1, < -1 -> -1, and if allow_short==False sets negative to 0.
        """
        import pandas as pd

        # If it's a pandas Series, use its values
        if isinstance(sig, pd.Series):
            try:
                arr = sig.to_numpy(dtype=int)
            except Exception as e:
                # try numeric conversion first
                arr = pd.to_numeric(sig, errors="coerce").fillna(0).astype(int).to_numpy()
        # If it's a pandas DataFrame, try to locate common column or fallback to single-col
        elif isinstance(sig, pd.DataFrame):
            # look for common column names used by strategies
            for col in ("signal", "signals", "position", "pos", "action"):
                if col in sig.columns:
                    try:
                        arr = sig[col].to_numpy(dtype=int)
                        break
                    except Exception:
                        arr = pd.to_numeric(sig[col], errors="coerce").fillna(0).astype(int).to_numpy()
                        break
            else:
                # fallback: if single-column DF, take that column
                if sig.shape[1] == 1:
                    try:
                        arr = sig.iloc[:, 0].to_numpy(dtype=int)
                    except Exception:
                        arr = pd.to_numeric(sig.iloc[:, 0], errors="coerce").fillna(0).astype(int).to_numpy()
                else:
                    raise ValueError("Strategy returned a DataFrame with multiple columns; cannot infer signal column. "
                                     "Return a 1-D array/Series or a DataFrame with a 'signal' column.")
        else:
            # fallback for sequences, numpy arrays, lists, generators, etc.
            try:
                arr = np.asarray(list(sig), dtype=int)
            except Exception:
                # last-resort: try to coerce to numeric with numpy
                try:
                    arr = np.asarray(sig, dtype=int)
                except Exception:
                    raise ValueError("Unable to convert signals to integer array. "
                                     "Signals must be iterable of ints, numpy array, pandas Series, or single-column DataFrame.")

        # enforce shape and values
        arr = np.asarray(arr, dtype=int)
        arr[arr > 1] = 1
        arr[arr < -1] = -1
        if not self.allow_short:
            arr[arr < 0] = 0
        return arr


    def run(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        datetime_col: Optional[str] = None,
        signals: Optional[Iterable[int]] = None,
        verbose: bool = False,
        order_unit_size: Optional[float] = None,
    ) -> Tuple[pd.Series, pd.DataFrame, Dict[str, float]]:
        if order_unit_size is None:
            order_unit_size = float(self.position_size)

        if isinstance(df, pd.Series):
            df = df.to_frame(name=price_col)
        if price_col not in df.columns:
            raise KeyError(f"Price column '{price_col}' not found in DataFrame")
        original_index = df.index

        # Reset index as before but keep mapping
        df_reset = df.reset_index(drop=False) if datetime_col else df.reset_index(drop=True)
        n = len(df_reset)
        if n == 0:
            raise ValueError("Empty dataframe passed to backtester")

        # ---- MICRO-OPT: extract price and volume arrays once ----
        price_arr = df_reset[price_col].to_numpy(dtype=float)
        if "volume" in df_reset.columns:
            vol_arr = df_reset["volume"].to_numpy(dtype=float)
        else:
            vol_arr = np.zeros(n, dtype=float)
        # ---------------------------------------------------------

        # signals
        if signals is None:
            if hasattr(self.strategy, "generate_signals") and callable(self.strategy.generate_signals):
                sig = self.strategy.generate_signals(df.copy())
            elif callable(self.strategy):
                sig = self.strategy(df.copy())
            else:
                raise ValueError("No signals provided and strategy is not callable or doesn't implement generate_signals(df)")
        else:
            sig = signals

        sig = self._normalize_signals(sig)
        if len(sig) != n:
            raise ValueError(f"Signals length {len(sig)} does not match data length {n}")

        # bookkeeping
        cash = float(self.initial_balance)
        position_units = 0.0
        equity_hist: List[float] = []
        index_hist: List[Any] = []
        order_queue: List[Dict[str, Any]] = []
        trades: List[Dict[str, Any]] = []
        total_commission = 0.0

        def apply_slippage_for_execution(raw_price: float, side: int, executed_fraction_of_liq: float, executed_units: float, bar_volume: float) -> float:
            base = abs(self.slippage_pct)
            if self.tca_params and float(self.tca_params.get("a", 0.0)) > 0.0 and bar_volume > 0.0:
                impact_frac = estimate_impact_from_params(executed_units, bar_volume, self.tca_params)
                impact = float(impact_frac)
            else:
                impact_coeff = max(base * 5.0, 0.001)
                impact = impact_coeff * (executed_fraction_of_liq ** 1.5)
            if side == 1:
                return float(raw_price * (1.0 + base + impact))
            else:
                return float(raw_price * (1.0 - base - impact))

        # main loop now iterates by index and uses price_arr/vol_arr
        for i in range(n):
            current_price = float(price_arr[i])
            bar_volume = float(vol_arr[i])

            # === Fix: when volume present use fraction only; fallback to order_unit_size when volume == 0
            if self.orderbook_sampler is None:
                if bar_volume > 0.0:
                    available_liquidity_value = float(bar_volume * self.liquidity_fraction)
                else:
                    available_liquidity_value = float(order_unit_size)
            else:
                available_liquidity_value = None

            s = int(sig[i])
            current_side = 0 if abs(position_units) < 1e-12 else (1 if position_units > 0 else -1)
            target_side = s

            # enqueue logic: closing order or change-of-side
            if target_side == 0 and current_side != 0:
                requested_closing = -position_units
                requested_abs = abs(requested_closing)
                if requested_abs > 1e-12:
                    closing_target = -current_side
                    order_queue.append({
                        "enqueue_idx": int(i),
                        "target": int(closing_target),
                        "requested_units": float(requested_closing),
                        "remaining_units": float(requested_abs),
                        "signed_remaining": float(requested_closing),
                    })
            elif target_side != 0 and target_side != current_side:
                requested_units = float(order_unit_size) * (1.0 if target_side == 1 else -1.0)
                delta_units = requested_units - position_units
                requested_abs = abs(delta_units)
                if requested_abs > 1e-12:
                    order_queue.append({
                        "enqueue_idx": int(i),
                        "target": int(target_side),
                        "requested_units": float(requested_units),
                        "remaining_units": float(requested_abs),
                        "signed_remaining": float(delta_units),
                    })

            remaining_queue: List[Dict[str, Any]] = []

            for order in order_queue:
                enqueue_idx = order["enqueue_idx"]
                if i < (enqueue_idx + self.fill_delay_steps):
                    remaining_queue.append(order)
                    continue

                needed_abs = float(order["remaining_units"])
                if needed_abs <= 0.0:
                    continue

                if self.orderbook_sampler is not None:
                    side = int(order["target"])
                    avail_side = float(self.orderbook_sampler.available_liquidity(i, side=side))
                    execable = min(needed_abs, avail_side)
                else:
                    execable = min(needed_abs, available_liquidity_value)

                if execable <= 0.0:
                    remaining_queue.append(order)
                    continue

                if self.orderbook_sampler is None:
                    denom = max(bar_volume * self.liquidity_fraction, execable, 1.0)
                else:
                    denom = max(float(bar_volume), execable, 1.0)

                executed_fraction = float(execable) / (denom + 1e-12)

                if self.orderbook_sampler is not None:
                    side = int(order["target"])
                    ob_res = self.orderbook_sampler.execute(i, side, execable)
                    exec_actual = float(ob_res["executed"])
                    if exec_actual <= 0.0:
                        remaining_queue.append(order)
                        continue
                    fill_price = float(ob_res["vwap"]) if ob_res.get("vwap", None) is not None else current_price
                    executed_units = exec_actual
                else:
                    executed_units = float(execable)
                    fill_price = apply_slippage_for_execution(current_price, int(order["target"]), executed_fraction, executed_units, bar_volume)

                sign = 1.0 if order["signed_remaining"] > 0.0 else -1.0
                executed_signed = executed_units * sign

                cash -= executed_signed * fill_price
                if abs(order["requested_units"]) > 0:
                    commission_charged = (executed_units / abs(order["requested_units"])) * self.commission
                else:
                    commission_charged = self.commission
                cash -= commission_charged
                total_commission += float(commission_charged)

                position_units += executed_signed

                order["remaining_units"] = max(0.0, order["remaining_units"] - executed_units)
                order["signed_remaining"] = (order["remaining_units"] if sign > 0.0 else -order["remaining_units"])

                trade = {
                    "enqueue_idx": int(enqueue_idx),
                    "fill_idx": int(i),
                    "enqueue_time": (original_index[enqueue_idx] if isinstance(original_index, pd.Index) else enqueue_idx),
                    "fill_time": (original_index[i] if isinstance(original_index, pd.Index) else i),
                    "side": int(order["target"]),
                    "fill_price": float(fill_price),
                    "requested_units": float(order["requested_units"]),
                    "executed_units": float(executed_units),
                    "remaining_units": float(order["remaining_units"]),
                    "commission": float(commission_charged),
                    "slippage": float(abs(fill_price - current_price)),
                    "cash_after": float(cash),
                    "position_after": float(position_units),
                }
                trades.append(trade)

                if order["remaining_units"] > 0.0:
                    remaining_queue.append(order)

            order_queue = remaining_queue

            unrealized = position_units * current_price
            equity = float(cash + unrealized)
            equity_hist.append(equity)
            try:
                idx_val = original_index[i]
            except Exception:
                idx_val = i
            index_hist.append(idx_val)

            if verbose:
                logger.debug("bar %s price=%.6f pos_units=%.4f cash=%.2f equity=%.2f orders=%d", i, current_price, position_units, cash, equity, len(order_queue))

        equity_series = pd.Series(data=np.array(equity_hist, dtype=float), index=pd.Index(index_hist))
        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            trades_df = pd.DataFrame(columns=[
                "enqueue_idx","fill_idx","enqueue_time","fill_time","side","fill_price",
                "requested_units","executed_units","remaining_units","commission","slippage",
                "cash_after","position_after"
            ])

        final_equity = float(equity_series.iloc[-1]) if len(equity_series) > 0 else float(cash)
        final_return = float(final_equity / self.initial_balance - 1.0) if self.initial_balance != 0 else 0.0
        returns = equity_series.pct_change().fillna(0.0).to_numpy(dtype=float) if len(equity_series) > 0 else np.array([], dtype=float)
        sharpe = _sharpe_from_returns(returns)
        max_dd = _max_drawdown_from_series(equity_series.to_numpy(dtype=float)) if len(equity_series) > 0 else 0.0

        metrics = {
            "final_equity": final_equity,
            "final_return": final_return,
            "total_trades": int(len(trades_df)),
            "total_commission": float(total_commission),
            "max_drawdown": float(max_dd),
            "sharpe": float(sharpe),
        }

        return equity_series, trades_df, metrics

