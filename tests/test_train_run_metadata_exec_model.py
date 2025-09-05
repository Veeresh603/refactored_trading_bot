# tests/test_train_run_metadata_exec_model.py
import json
import glob
import os
from ai.train_rl import TrainConfig, RLTrainer

def test_run_metadata_contains_execution_model(tmp_path):
    # small config: short run, synthetic env is used by trainer._make_env
    cfg = TrainConfig(episodes=2, steps_per_episode=5, checkpoint_dir=str(tmp_path), seed=42)
    trainer = RLTrainer(cfg)
    # run a tiny training loop (will create run_metadata file)
    trainer.train()
    # find run metadata
    files = glob.glob(os.path.join(str(tmp_path), "run_metadata_*.json"))
    assert len(files) == 1, f"No run_metadata file found in {str(tmp_path)}"
    with open(files[0], "r") as f:
        data = json.load(f)
    assert "execution_model" in data
    em = data["execution_model"]
    # fields exist (may be None if env doesn't expose them)
    assert set(["fill_delay_steps", "slippage_pct", "commission"]).issubset(set(em.keys()))
