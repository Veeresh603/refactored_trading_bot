"""
RL Allocator Callback
---------------------
- Tracks Sharpe & Max Drawdown during PPO training
- Logs metrics to TensorBoard
- Auto-saves best checkpoint
- Optional Telegram alerts
"""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from backtesting.metrics import _calc_sharpe, _calc_max_drawdown
from core.utils import send_telegram_message


class AllocatorMetricsCallback(BaseCallback):
    def __init__(self, save_path="checkpoints", verbose=0, telegram=False):
        super(AllocatorMetricsCallback, self).__init__(verbose)
        self.returns = []
        self.equity_curve = []
        self.best_sharpe = -np.inf
        self.save_path = save_path
        self.telegram = telegram
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Log reward as return
        reward = self.locals.get("rewards", [0])[-1]
        self.returns.append(reward)

        # Track equity (cumulative)
        equity = sum(self.returns)
        self.equity_curve.append(equity)

        # Compute metrics
        sharpe = _calc_sharpe(np.array(self.returns)) if len(self.returns) > 2 else 0
        dd = _calc_max_drawdown(np.array(self.equity_curve)) if len(self.equity_curve) > 2 else 0

        # Log to TensorBoard
        self.logger.record("allocator/sharpe", sharpe)
        self.logger.record("allocator/max_drawdown", dd)

        # Save best model
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            save_file = os.path.join(self.save_path, "best_model")
            self.model.save(save_file)
            if self.verbose > 0:
                print(f"💾 New best model saved @ {save_file} (Sharpe={sharpe:.2f})")
            if self.telegram:
                send_telegram_message(f"🤖 New best RL Allocator model! Sharpe={sharpe:.2f}")

        return True
