# make_and_run_supervised.py
import numpy as np
import os
from ai.train_rl import TrainConfig, RLTrainer

out_dir = "data"
os.makedirs(out_dir, exist_ok=True)

# 1) Create toy features (N samples x F features)
N = 200
F = 8
rng = np.random.RandomState(42)
features = rng.normal(size=(N, F)).astype(float)
labels = (rng.rand(N) > 0.5).astype(float)  # binary labels example

feat_path = os.path.join(out_dir, "supervised_features.npy")
lab_path = os.path.join(out_dir, "supervised_labels.npy")
np.save(feat_path, features)
np.save(lab_path, labels)
print("Wrote:", feat_path, lab_path)

# 2) Create features_meta.json (optional but useful)
import json
meta = {
    "n_features": F,
    "feature_names": [f"f{i}" for i in range(F)],
    "source": "synthetic_example",
    "created_by": "make_and_run_supervised.py"
}
with open(os.path.join(out_dir, "features_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("Wrote features_meta.json")

# 3) Run a short training with supervised paths in TrainConfig
ckpt_dir = "checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
cfg = TrainConfig(
    episodes=2,
    steps_per_episode=50,
    checkpoint_dir=ckpt_dir,
    seed=123,
    supervised_features=feat_path,
    supervised_labels=lab_path,
    val_n_splits=3
)

trainer = RLTrainer(cfg)
trainer.train()

print("Trainer finished. Check:", ckpt_dir)
