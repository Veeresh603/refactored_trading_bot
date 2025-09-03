"""
Strategy Engine
---------------
- Runs multiple strategies in parallel
- Collects signals into a consolidated decision
- RLAllocator can act as meta-strategy
"""

import logging
import pandas as pd
from typing import List, Dict, Any
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger("StrategyEngine")


class StrategyEngine:
    def __init__(self, strategies: List[BaseStrategy]):
        """
        Args:
            strategies (list): List of strategy instances (BaseStrategy subclasses or RLAllocator)
        """
        self.strategies = strategies

    def run(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all strategies and return consolidated decision.

        Args:
            market_data (dict): must contain
                - asset (str)
                - spot (float)
                - history (list/Series of past closes)

        Returns:
            dict with keys:
                - strategy (name)
                - signal (int)
                - strike_offset (int)
                - expiry (str)
        """
        asset = market_data.get("asset")
        spot = market_data.get("spot")
        history = pd.Series(market_data.get("history", []))

        decisions = {}

        # --- Run each strategy ---
        for strat in self.strategies:
            try:
                if hasattr(strat, "choose_action"):  # RLAllocator style
                    decision = strat.choose_action()
                    decisions[strat.__class__.__name__] = decision
                else:  # Classical BaseStrategy
                    df = pd.DataFrame({"time": range(len(history)), "close": history})
                    signals = strat.generate_signals(df)
                    signal = signals["signal"].iloc[-1] if not signals.empty else 0
                    decisions[strat.name] = {"strategy": strat.name, "signal": int(signal)}
            except Exception as e:
                logger.error(f"❌ Strategy {strat.__class__.__name__} failed: {e}")

        # --- Consolidation ---
        final_decision = self._consolidate(decisions, spot)
        logger.info(f"📊 Consolidated decision: {final_decision}")
        return final_decision

    def _consolidate(self, decisions: Dict[str, Any], spot: float) -> Dict[str, Any]:
        """
        Combine multiple strategy outputs into one decision.
        RLAllocator acts as tie-breaker / allocator if present.
        """
        # If RLAllocator present, trust it
        if "RLAllocator" in decisions:
            return decisions["RLAllocator"]

        # Otherwise, simple majority vote among classical strategies
        votes = [d["signal"] for d in decisions.values() if "signal" in d]
        final_signal = int(round(sum(votes) / len(votes))) if votes else 0

        return {
            "strategy": "Ensemble",
            "signal": final_signal,
            "strike_offset": 0,
            "expiry": "weekly",
        }
