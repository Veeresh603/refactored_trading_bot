# core/paper_engine.py
"""
Paper Trading Engine
--------------------
- Simulates live trading using option *premiums* (not index spot) for PnL
- Maintains a FIFO trade ledger in *lots*; converts to contracts for C++ engine
- Netted open positions view (no duplicated +/− spam rows, no zero-qty)
- Premium-marked UPL via Black–Scholes; realized PnL via FIFO matching
- Forwards execution and Greeks to the C++ execution_engine
"""

from __future__ import annotations

import math
import logging
from datetime import datetime
from collections import deque, defaultdict
from typing import Dict, List, Optional

from core import execution_engine_cpp as execution_engine  # pybind11 module exposing class ExecutionEngine

logger = logging.getLogger("PaperEngine")


# ------------------------
# Black–Scholes helpers
# ------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(spot: float, strike: float, T: float, sigma: float, is_call: bool) -> float:
    """Returns the *premium* (option price), not the index level."""
    if T <= 0.0 or sigma <= 0.0 or spot <= 0.0 or strike <= 0.0:
        return 0.0
    d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return spot * _norm_cdf(d1) - strike * _norm_cdf(d2)
    else:
        return strike * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


class PaperTradingEngine:
    """
    Public API used by main.py:
      - reset()
      - place_order(symbol, qty, price, side="BUY", strike=None, sigma=0.2, is_call=True, expiry_days=30)
      - account_status(spot_price)
      - portfolio_greeks(spot_price)
      - list_open_positions(with_pnl: bool = False, spot_price: Optional[float] = None)
    """

    def __init__(self, initial_balance: float = 100_000.0, lot_size: int = 50):
        self.initial_balance = float(initial_balance)
        self.lot_size = int(lot_size)

        # pybind11 C++ engine (expects *contracts*, not lots)
        try:
            self.engine = execution_engine.ExecutionEngine()
        except Exception:
            # Some builds expose functions at module level; fallback to module
            self.engine = execution_engine

        # Trade ledger: list of dicts in *lots* (signed). entry_price is *premium*.
        self.trades: List[Dict] = []

        self.reset()
        logger.info(f"📝 PaperTradingEngine initialized with balance={self.initial_balance}, lot_size={self.lot_size}")

    # ------------------------
    # Lifecycle
    # ------------------------
    def reset(self):
        """Reset C++ engine and local ledger."""
        try:
            # Class API
            self.engine.reset(self.initial_balance)
        except AttributeError:
            # Module API fallback (rare)
            if hasattr(self.engine, "reset_engine"):
                self.engine.reset_engine(self.initial_balance)
            elif hasattr(self.engine, "reset"):
                self.engine.reset(self.initial_balance)
            else:
                logger.warning("⚠️ execution_engine has no reset method; continuing without engine reset")

        self.trades.clear()
        logger.info("🔄 Paper engine reset")

    # ------------------------
    # Trading
    # ------------------------
    def place_order(
        self,
        symbol: str,
        qty: int,
        price: float,
        side: str = "BUY",
        strike: Optional[float] = None,
        sigma: float = 0.20,
        is_call: bool = True,
        expiry_days: int = 30,
    ):
        """
        Place a simulated order.

        Parameters
        ----------
        symbol : str       Option symbol, e.g. 'NIFTY20000CE'
        qty    : int       *Lots* (signed handled by side). 0 is ignored.
        price  : float     Option *premium* per contract
        side   : str       'BUY' or 'SELL'
        strike : float     Strike; if None, attempts to parse from symbol (optional upstream)
        sigma  : float     Implied vol (annualized)
        is_call: bool      Call vs Put
        expiry_days : int  Days to expiry; clamped to >= 1
        """
        if qty == 0:
            return  # ignore zero-qty noise

        side_up = side.upper()
        signed_lots = int(qty) if side_up == "BUY" else -int(qty)
        contracts = signed_lots * self.lot_size
        safe_expiry = int(max(1, expiry_days))
        safe_sigma = float(max(1e-6, sigma))  # avoid zero vol

        # Persist to *lots* ledger with premium entry
        self.trades.append(
            {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "lots": signed_lots,           # in lots
                "entry_price": float(price),   # premium per contract
                "strike": float(strike or 0.0),
                "sigma": safe_sigma,
                "is_call": bool(is_call),
                "expiry_days": safe_expiry,
            }
        )

        # Forward to C++ engine in *contracts*
        try:
            # Expecting: (symbol, qty_contracts, price, strike, sigma, is_call, expiry_days)
            self.engine.place_order(
                symbol,
                int(contracts),
                float(price),
                float(strike or 0.0),
                safe_sigma,
                bool(is_call),
                float(safe_expiry),
            )
            side_verb = "BUY" if signed_lots > 0 else "SELL"
            logger.info(
                f"📝 PAPER {side_verb} {symbol} x{abs(qty)} lots "
                f"({contracts:+d} contracts) @ {price:.2f}"
            )
        except Exception as e:
            logger.error(f"❌ Paper order failed: {e}")

    # ------------------------
    # Positions (netted, premium-marked)
    # ------------------------
    def list_open_positions(self, with_pnl: bool = False, spot_price: Optional[float] = None) -> List[Dict]:
        """
        Returns a CLEAN, NETTED view of open positions by symbol.

        lots: signed lots (positive = long, negative = short)
        qty:  contracts = lots * lot_size
        entry_price: VWAP of remaining open side (premium)
        If with_pnl=True and spot_price provided → computes UPL via Black–Scholes mark.
        Realized PnL is aggregated via FIFO matching for each symbol.
        """
        long_fifo: Dict[str, deque] = defaultdict(deque)   # symbol -> deque[(lots>0, prem)]
        short_fifo: Dict[str, deque] = defaultdict(deque)  # symbol -> deque[(lots<0, prem)]
        realized_by_symbol: Dict[str, float] = defaultdict(float)

        # Build FIFO queues and compute realized on the fly (premium-based, per contract)
        for tr in self.trades:
            sym = tr["symbol"]
            lots = int(tr["lots"])                   # signed lots (+ buy, − sell)
            eprem = float(tr["entry_price"])         # premium per contract

            if lots == 0:
                continue

            if lots > 0:  # BUY → offset shorts first
                remain = lots
                while remain > 0 and short_fifo[sym]:
                    s_lots, s_ep = short_fifo[sym][0]   # s_lots is negative
                    match = min(remain, -s_lots)
                    # Realized = (short_entry - buy_close) * contracts
                    realized_by_symbol[sym] += (s_ep - eprem) * match * self.lot_size
                    s_lots += match
                    remain -= match
                    if s_lots == 0:
                        short_fifo[sym].popleft()
                    else:
                        short_fifo[sym][0] = (s_lots, s_ep)
                if remain > 0:  # leftover → open long
                    long_fifo[sym].append((remain, eprem))

            else:  # SELL → offset longs first
                remain = -lots
                while remain > 0 and long_fifo[sym]:
                    l_lots, l_ep = long_fifo[sym][0]
                    match = min(remain, l_lots)
                    # Realized = (sell_entry - long_entry) * contracts
                    realized_by_symbol[sym] += (eprem - l_ep) * match * self.lot_size
                    l_lots -= match
                    remain -= match
                    if l_lots == 0:
                        long_fifo[sym].popleft()
                    else:
                        long_fifo[sym][0] = (l_lots, l_ep)
                if remain > 0:  # leftover → open short (store negative)
                    short_fifo[sym].append((-remain, eprem))

        rows: List[Dict] = []
        symbols = sorted(set(long_fifo.keys()) | set(short_fifo.keys()))
        for sym in symbols:
            net_lots = sum(l for l, _ in long_fifo[sym]) + sum(l for l, _ in short_fifo[sym])  # short lots negative
            if net_lots == 0:
                continue

            # VWAP of remaining side
            if net_lots > 0:  # remaining longs
                total_lots = sum(l for l, _ in long_fifo[sym])
                vwap = sum(l * ep for l, ep in long_fifo[sym]) / total_lots if total_lots > 0 else 0.0
            else:  # remaining shorts (store as negative lots)
                total_lots = -sum(l for l, _ in short_fifo[sym])
                vwap = sum((-l) * ep for l, ep in short_fifo[sym]) / total_lots if total_lots > 0 else 0.0

            # Pull metadata from the most recent trade for that symbol
            sample = next((t for t in reversed(self.trades) if t["symbol"] == sym), None)
            is_call = sym.endswith("CE")
            strike = float(sample["strike"]) if sample else 0.0
            sigma = float(sample["sigma"]) if sample else 0.20
            days = int(sample["expiry_days"]) if sample else 7

            row = {
                "symbol": sym,
                "lots": int(net_lots),
                "qty": int(net_lots * self.lot_size),
                "strike": strike,
                "sigma": sigma,
                "expiry_days": max(1, days),
                "entry_price": vwap,
                "realized_pnl": realized_by_symbol[sym],
            }

            if with_pnl and spot_price is not None:
                T = row["expiry_days"] / 365.0
                mark = _bs_price(float(spot_price), strike, T, sigma, is_call)
                row["unrealized_pnl"] = (mark - vwap) * row["qty"]

            rows.append(row)

        return rows

    # ------------------------
    # Account + Greeks
    # ------------------------
    def account_status(self, spot_price: float) -> Dict[str, float]:
        """
        Premium-based account snapshot:
          - realized: FIFO PnL from matched premium differences
          - unrealized: mark-to-market via Black–Scholes premium
          - total: initial_balance + realized + unrealized
        """
        open_rows = self.list_open_positions(with_pnl=True, spot_price=spot_price)
        realized = float(sum(r.get("realized_pnl", 0.0) for r in open_rows))
        unrealized = float(sum(r.get("unrealized_pnl", 0.0) for r in open_rows))
        total = self.initial_balance + realized + unrealized
        return {
            "balance": self.initial_balance,  # keep static in paper mode (or update if you model cash)
            "realized": realized,
            "unrealized": unrealized,
            "total": total,
        }

    def portfolio_greeks(self, spot_price: float):
        """
        Forward to C++ engine for Greeks (contracts-based).
        """
        try:
            return self.engine.portfolio_greeks(float(spot_price))
        except Exception as e:
            logger.error(f"❌ portfolio_greeks failed: {e}")
            return (0.0, 0.0, 0.0, 0.0)

    # ------------------------
    # Utilities
    # ------------------------
    def get_trade_log(self) -> List[Dict]:
        """Return the raw trade ledger (in lots)."""
        return list(self.trades)
