# core/circuit_breaker.py
class CircuitBreaker:
    def __init__(self, max_daily_loss=5000, max_drawdown=0.2, max_consecutive_losses=5):
        """
        Trading safety circuit breaker.

        Args:
            max_daily_loss (float): Max allowed daily loss (₹ or $)
            max_drawdown (float): Max allowed equity drawdown (fraction, e.g., 0.2 = 20%)
            max_consecutive_losses (int): Max consecutive losing trades before stop
        """
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.max_consecutive_losses = max_consecutive_losses

        self.daily_loss = 0
        self.max_equity = 0
        self.consecutive_losses = 0

    def check(self, equity, trade_pnl):
        """
        Check trading health.

        Args:
            equity (float): Current account equity
            trade_pnl (float): Profit/loss from last trade

        Returns:
            str: "OK" or STOP reason
        """
        # Track cumulative daily losses
        if trade_pnl < 0:
            self.daily_loss += trade_pnl
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        self.max_equity = max(self.max_equity, equity)

        if self.daily_loss < -self.max_daily_loss:
            return "STOP: Daily loss limit exceeded"
        if equity < self.max_equity * (1 - self.max_drawdown):
            return "STOP: Drawdown limit exceeded"
        if self.consecutive_losses >= self.max_consecutive_losses:
            return "STOP: Too many consecutive losses"

        return "OK"
