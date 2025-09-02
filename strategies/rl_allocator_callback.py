import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class AllocatorMetricsCallback(BaseCallback):
    """
    PPO Callback for:
    - Logging Sharpe & Max Drawdown to TensorBoard
    - Auto-saving best model checkpoints
    """

    def __init__(self, save_path="checkpoints", verbose=0):
        super(AllocatorMetricsCallback, self).__init__(verbose)
        self.returns = []
        self.equity_curve = []
        self.best_sharpe = -np.inf
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if "returns" in self.locals:
            step_returns = self.locals["returns"]
            if isinstance(step_returns, (list, np.ndarray)):
                self.returns.extend(step_returns)

        if len(self.returns) > 20:
            mean_ret = np.mean(self.returns)
            std_ret = np.std(self.returns) + 1e-8
            sharpe = mean_ret / std_ret

            equity = (self.equity_curve[-1] if self.equity_curve else 1.0) * (1 + self.returns[-1])
            self.equity_curve.append(equity)
            peak = np.maximum.accumulate(self.equity_curve)
            drawdowns = (self.equity_curve - peak) / peak
            max_dd = drawdowns.min()

            # Log metrics
            self.logger.record("allocator/sharpe", sharpe)
            self.logger.record("allocator/max_drawdown", max_dd)

            # Checkpoint if Sharpe improved
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                checkpoint_path = os.path.join(self.save_path, f"allocator_best_{self.num_timesteps}.zip")
                self.model.save(checkpoint_path)
                if self.verbose > 0:
                    print(f"💾 New best Sharpe {sharpe:.2f} at step {self.num_timesteps}, saved {checkpoint_path}")

        return True
