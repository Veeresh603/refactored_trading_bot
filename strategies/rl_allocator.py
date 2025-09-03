import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class AllocatorMetricsCallback(BaseCallback):
    """
    PPO Callback for RL Allocator:
    - Logs Sharpe ratio & Max Drawdown
    - Logs equity, drawdown, margin usage to TensorBoard
    - Auto-saves best model checkpoints
    """

    def __init__(self, save_path="checkpoints", verbose=0):
        super(AllocatorMetricsCallback, self).__init__(verbose)
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        # Tracking
        self.returns = []
        self.equity_curve = []
        self.drawdown_history = []
        self.margin_history = []

        self.best_sharpe = -np.inf
        self.peak_equity = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        for info in infos:
            if "equity" in info:
                equity = float(info["equity"])
                margin = float(info.get("margin_used", 0))

                self.equity_curve.append(equity)
                self.margin_history.append(margin)

                # Peak equity + drawdown tracking
                if self.peak_equity is None:
                    self.peak_equity = equity
                self.peak_equity = max(self.peak_equity, equity)
                drawdown = self.peak_equity - equity
                self.drawdown_history.append(drawdown)

                # Compute rolling returns for Sharpe
                if len(self.equity_curve) > 1:
                    ret = equity - self.equity_curve[-2]
                    self.returns.append(ret)

                # Log to TensorBoard
                self.logger.record("custom/equity", equity)
                self.logger.record("custom/drawdown", drawdown)
                self.logger.record("custom/margin", margin)

        # Periodically evaluate performance
        if len(self.returns) > 30:  # after ~30 steps
            sharpe = self._compute_sharpe(self.returns)
            max_dd = max(self.drawdown_history) if self.drawdown_history else 0

            self.logger.record("custom/sharpe", sharpe)
            self.logger.record("custom/max_drawdown", max_dd)

            # Save best checkpoint
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                model_path = os.path.join(self.save_path, "best_model")
                self.model.save(model_path)
                if self.verbose > 0:
                    print(f"💾 New best model saved at {model_path} (Sharpe={sharpe:.2f})")

        return True

    def _compute_sharpe(self, returns, risk_free=0.0):
        returns = np.array(returns)
        if returns.std() == 0:
            return -np.inf
        return (returns.mean() - risk_free) / (returns.std() + 1e-8)

    def _on_training_end(self) -> None:
        if self.verbose > 0 and self.equity_curve:
            final_equity = self.equity_curve[-1]
            max_dd = max(self.drawdown_history) if self.drawdown_history else 0
            print(f"📊 Training complete | Final Equity={final_equity:.2f} | MaxDD={max_dd:.2f}")
