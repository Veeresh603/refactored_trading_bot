"""
Advanced Risk Manager
---------------------
Centralized risk checks for trading bot.

Checks:
- Max position size
- Max daily loss
- Max drawdown
- Max consecutive losses
- Circuit-breaker (stop trading for the day)
"""

import logging
from datetime import datetime, timedelta

logger = logging.getLogger("RiskManager")


class AdvancedRiskManager:
    def __init__(self, config):
        """
        Args:
            config (dict): from config.yaml "risk"
        """
        self.max_position_size = config.get("max_position_size", 100)
        self.max_daily_loss = config.get("max_daily_loss", 5000)
        self.max_drawdown = config.get("max_drawdown", 10000)
        self.max_consecutive_losses = config.get("max_consecutive_losses", 3)
        self.stop_on_loss = config.get("stop_on_loss", True)
        self.cool_off_minutes = config.get("cool_off_minutes", 30)

        # State
        self.daily_start = datetime.now().date()
        self.daily_loss = 0
        self.peak_equity = None
        self.consecutive_losses = 0
        self.cool_off_until = None

    def reset_daily(self):
        """Reset daily counters at new trading day."""
        self.daily_start = datetime.now().date()
        self.daily_loss = 0
        self.peak_equity = None
        self.consecutive_losses = 0
        self.cool_off_until = None
        logger.info("🔄 Risk manager daily reset")

    def check_risk(self, equity, pnl, position_size):
        """
        Check all risk conditions.

        Args:
            equity (float): current equity
            pnl (float): latest realized/unrealized PnL
            position_size (int): current position size

        Returns:
            (bool, str) -> (is_safe, reason_if_not)
        """
        now = datetime.now()

        # Reset at new day
        if now.date() != self.daily_start:
            self.reset_daily()

        # Cool-off period
        if self.cool_off_until and now < self.cool_off_until:
            return False, f"Cool-off active until {self.cool_off_until.strftime('%H:%M:%S')}"

        # Peak equity tracking
        if self.peak_equity is None:
            self.peak_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = self.peak_equity - equity

        # --- Checks ---
        if position_size > self.max_position_size:
            return False, f"Position size {position_size} > max {self.max_position_size}"

        if abs(self.daily_loss + pnl) > self.max_daily_loss:
            if self.stop_on_loss:
                return False, "Max daily loss breached - STOP"
            else:
                self.cool_off_until = now + timedelta(minutes=self.cool_off_minutes)
                return False, "Max daily loss breached - COOL OFF"

        if drawdown > self.max_drawdown:
            if self.stop_on_loss:
                return False, "Max drawdown breached - STOP"
            else:
                self.cool_off_until = now + timedelta(minutes=self.cool_off_minutes)
                return False, "Max drawdown breached - COOL OFF"

        return True, "OK"

    def update_loss_tracking(self, pnl_change):
        """
        Update consecutive losses counter.
        """
        if pnl_change < 0:
            self.consecutive_losses += 1
            logger.warning(f"⚠️ Consecutive loss {self.consecutive_losses}")
            if self.consecutive_losses >= self.max_consecutive_losses:
                if self.stop_on_loss:
                    return False, "Max consecutive losses reached - STOP"
                else:
                    self.cool_off_until = datetime.now() + timedelta(minutes=self.cool_off_minutes)
                    self.consecutive_losses = 0
                    return False, "Max consecutive losses reached - COOL OFF"
        else:
            self.consecutive_losses = 0
        return True, "OK"
