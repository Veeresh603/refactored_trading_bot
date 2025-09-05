# ai/rl_agent.py
"""
Simple REINFORCE-style RLAgent (NumPy) for discrete action spaces.

Features:
- Constructor accepts lr and gamma: RLAgent(obs_dim, act_dim, lr=1e-3, gamma=0.99)
- select_action(obs): returns action (int) and stores log-prob for learning
- store_transition(obs, action, reward): (optional) you can store external transitions
- update(): runs REINFORCE update on stored episode, clears buffer
- save(path) / load(path): persist policy weights using pickle

This is intentionally small and dependency-free so it can be used for quick
experimentation and training in the repo you showed me.
"""

from __future__ import annotations
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


class RLAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        seed: Optional[int] = None,
    ):
        """
        Args:
            obs_dim: dimensionality of observation vector (int)
            act_dim: number of discrete actions (int)
            lr: learning rate for policy updates (float)
            gamma: discount factor (float)
            seed: optional random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.lr = float(lr)
        self.gamma = float(gamma)

        # Policy parameters: simple linear mapping obs -> logits
        # shape: (obs_dim, act_dim)
        # small random init
        self.W = np.random.randn(self.obs_dim, self.act_dim) * 0.01
        self.b = np.zeros((self.act_dim,), dtype=float)

        # Episode buffer for REINFORCE
        self._obs_buf: List[np.ndarray] = []
        self._act_buf: List[int] = []
        self._rew_buf: List[float] = []

    # --------------------------
    # Interaction methods
    # --------------------------
    def _policy(self, obs: np.ndarray) -> np.ndarray:
        """Return action probabilities for a single observation."""
        obs = np.asarray(obs, dtype=float)
        if obs.ndim == 1 and obs.shape[0] != self.obs_dim:
            raise ValueError(f"obs shape mismatch: got {obs.shape}, expected ({self.obs_dim},)")
        logits = obs.dot(self.W) + self.b  # (act_dim,)
        probs = softmax(logits)
        return probs

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        Choose an action given observation.
        - deterministic=True -> argmax policy (useful for evaluation)
        - deterministic=False -> sample according to softmax probs
        Returns:
            action (int)
        """
        probs = self._policy(obs)
        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(self.act_dim, p=probs))
        return action

    def store_transition(self, obs: np.ndarray, action: int, reward: float):
        """Append a transition to the internal episode buffer."""
        self._obs_buf.append(np.asarray(obs, dtype=float).copy())
        self._act_buf.append(int(action))
        self._rew_buf.append(float(reward))

    # --------------------------
    # Learning / update
    # --------------------------
    def _discounted_returns(self, rewards: List[float]) -> np.ndarray:
        """Compute discounted returns (simple, no baseline)."""
        R = np.zeros_like(rewards, dtype=float)
        running = 0.0
        for i in reversed(range(len(rewards))):
            running = rewards[i] + self.gamma * running
            R[i] = running
        # Normalize returns for stability
        if len(R) > 1:
            R = (R - R.mean()) / (R.std() + 1e-8)
        return R

    def update(self):
        """
        Perform a REINFORCE policy gradient update using the stored episode.
        Clears buffers after update.
        """
        if len(self._rew_buf) == 0:
            return  # nothing to update

        obs = np.vstack(self._obs_buf)         # (T, obs_dim)
        acts = np.asarray(self._act_buf, int)  # (T,)
        rets = self._discounted_returns(self._rew_buf)  # (T,)

        # Compute action probs and gradients
        logits = obs.dot(self.W) + self.b               # (T, act_dim)
        probs = softmax(logits)                         # (T, act_dim)

        # Create one-hot for actions
        T = probs.shape[0]
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(T), acts] = 1.0

        # Policy gradient: grad log pi(a|s) = (one_hot - probs) w.r.t logits
        # dL/dW = - sum_t [ grad log pi * R_t * s_t^T ]
        # We'll perform a simple gradient step (gradient ascent on expected return)
        advantages = rets  # no baseline here
        # Expand advantages to match shape (T, 1)
        adv = advantages.reshape(-1, 1)

        # gradient w.r.t logits: (one_hot - probs) * adv (we want ascent)
        grad_logits = (one_hot - probs) * adv  # (T, act_dim)

        # gradient w.r.t weights: obs.T @ grad_logits
        grad_W = obs.T.dot(grad_logits) / float(T)
        grad_b = np.mean(grad_logits, axis=0)

        # Gradient ascent step (we want to maximize expected return)
        self.W += self.lr * grad_W
        self.b += self.lr * grad_b

        # Clear buffers
        self._obs_buf.clear()
        self._act_buf.clear()
        self._rew_buf.clear()

    # --------------------------
    # Persistence
    # --------------------------
    def save(self, path: str):
        """Save model parameters to a pickle file."""
        base_dir = os.path.dirname(path)
        if base_dir and not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        payload = {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "lr": self.lr,
            "gamma": self.gamma,
            "W": self.W,
            "b": self.b,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        return path

    @classmethod
    def load(cls, path: str) -> "RLAgent":
        """Load an agent from disk."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        agent = cls(payload["obs_dim"], payload["act_dim"], lr=payload.get("lr", 1e-3), gamma=payload.get("gamma", 0.99))
        agent.W = payload["W"]
        agent.b = payload["b"]
        return agent

    # --------------------------
    # Utilities for training loop
    # --------------------------
    def reset_buffers(self):
        """Clear internal buffers without updating."""
        self._obs_buf.clear()
        self._act_buf.clear()
        self._rew_buf.clear()

    def get_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (W, b) for inspection / debugging."""
        return self.W.copy(), self.b.copy()
