import time
import logging

logger = logging.getLogger("RiskManager")


class AdvancedRiskManager:
    def __init__(self, config):
        self.config = config
        self.max_position_size = config.get("max_position_size", 50)
        self.max_daily_loss = config.get("max_daily_loss", -20000)
        self.max_drawdown = config.get("max_drawdown", -100000)
        self.max_consecutive_losses = config.get("max_consecutive_losses", 5)
        self.stop_on_loss = config.get("stop_on_loss", True)
        self.cool_off_minutes = config.get("cool_off_minutes", 15)
        self.max_short_lots = config.get("max_short_lots", 0)  # 🚀 NEW

        # State
        self.daily_pnl = 0.0
        self.drawdown = 0.0
        self.consecutive_losses = 0
        self.equity_peak = 0.0
        self.cooloff_until = 0
        self.cooloff_active = False

        logger.info("⚙️ AdvancedRiskManager initialized")

    # -------------------------------
    # PnL tracking
    # -------------------------------
    def update_pnl(self, realized, unrealized):
        total = realized + unrealized
        self.daily_pnl = total
        if total > self.equity_peak:
            self.equity_peak = total
        self.drawdown = total - self.equity_peak

    # -------------------------------
    # Main risk check
    # -------------------------------
    def check_risk(self, equity, pnl, position_size, delta_exposure=0, new_trade_delta=0, current_short_lots=0):
        # 1) Max position size
        if position_size > self.max_position_size:
            return False, f"Position size {position_size} > max {self.max_position_size}"

        # 2) Max daily loss
        if self.daily_pnl <= self.max_daily_loss:
            return False, "Max daily loss breached"

        # 3) Max drawdown
        if self.drawdown <= self.max_drawdown:
            return False, "Max drawdown breached"

        # 4) Short exposure cap 🚀
        if new_trade_delta < 0:  # SELL trade
            if current_short_lots >= self.max_short_lots:
                return False, f"Short exposure capped: {current_short_lots}/{self.max_short_lots} lots"

        return True, "OK"

    # -------------------------------
    # Cool-off logic
    # -------------------------------
    def start_cooloff(self, delta_exposure=0, new_trade_delta=0):
        """Start cool-off. Can be smarter: allow hedges, dynamic length."""
        length = self.cool_off_minutes * 60

        # Dynamic cool-off: if high volatility, double it
        if abs(delta_exposure) > self.max_position_size / 2:
            length *= 2

        self.cooloff_until = time.time() + length
        self.cooloff_active = True
        logger.warning(f"⏸️ Cool-off started for {length//60} minutes")

    def in_cooloff(self):
        if self.cooloff_active and time.time() < self.cooloff_until:
            return True
        if self.cooloff_active and time.time() >= self.cooloff_until:
            self.cooloff_active = False
        return False

    # -------------------------------
    # Utility
    # -------------------------------
    def allowed_risk_per_trade(self, equity):
        return equity * self.config.get("risk_per_trade", 0.01)
