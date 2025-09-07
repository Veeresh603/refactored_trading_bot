# tests/test_train_recovers_from_env_errors.py
import os
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from ai.train_rl import TrainConfig, RLTrainer


class FlakyEnv(SimpleNamespace):
    """
    Minimal fake env: raises on step when idx in bad_steps,
    otherwise returns obs, reward, done, info in gym-style.
    """
    def __init__(self, window=3, episode_length=10, bad_steps=None):
        self.window = window
        self.episode_length = episode_length
        self._step = 0
        self.bad_steps = set(bad_steps or [])
        self.prices = [100.0 + i for i in range(episode_length + window)]
        self.position = 0.0

    def reset(self):
        self._step = 0
        return [0.0] * self.window

    def step(self, action):
        # raise on configured steps to simulate crashy sampler/env
        if self._step in self.bad_steps:
            self._step += 1
            raise RuntimeError(f"simulated env error at step {self._step - 1}")
        # simple deterministic step
        obs = [float(self._step + i) for i in range(self.window)]
        reward = 0.0
        done = (self._step >= self.episode_length - 1)
        info = {"ordered": False, "fill": False, "fill_price": None, "executed_units": 0.0, "liquidity_used": 0.0}
        self._step += 1
        return obs, reward, done, info


def test_trainer_recovers_from_env_step_exceptions(tmp_path, monkeypatch):
    ckpt_dir = str(tmp_path / "checkpoints")
    cfg = TrainConfig(episodes=1, steps_per_episode=8, checkpoint_dir=ckpt_dir, seed=123)
    trainer = RLTrainer(cfg)

    # monkeypatch _make_env to return our flaky env
    flaky = FlakyEnv(window=3, episode_length=8, bad_steps={1, 4})
    monkeypatch.setattr(trainer, "_make_env", lambda: flaky)

    # Should not raise
    trainer.train()

    # Ensure run metadata file written
    files = list(Path(ckpt_dir).glob("run_metadata_*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert "run_id" in data
    # trainer should have finished
    assert "finished_at" in data
