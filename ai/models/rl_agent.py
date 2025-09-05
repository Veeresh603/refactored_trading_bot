# ai/models/rl_agent.py
"""
Small MLP policy gradient agent with optional value baseline.
"""
from __future__ import annotations
import numpy as np
import pickle
from typing import List, Optional, Tuple


def relu(x): return np.maximum(0.0, x)
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


class MLPAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        seed: Optional[int] = None,
        use_baseline: bool = True,
    ):
        if seed is not None:
            np.random.seed(seed)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.use_baseline = bool(use_baseline)

        self.W1 = np.random.randn(self.obs_dim, self.hidden) * 0.01
        self.b1 = np.zeros((self.hidden,), dtype=float)
        self.W2 = np.random.randn(self.hidden, self.act_dim) * 0.01
        self.b2 = np.zeros((self.act_dim,), dtype=float)

        if self.use_baseline:
            self.Wv1 = np.random.randn(self.obs_dim, self.hidden) * 0.01
            self.bv1 = np.zeros((self.hidden,), dtype=float)
            self.Wv2 = np.random.randn(self.hidden, 1) * 0.01
            self.bv2 = np.zeros((1,), dtype=float)

        self._obs_buf: List[np.ndarray] = []
        self._act_buf: List[int] = []
        self._rew_buf: List[float] = []

    def _policy_forward(self, obs: np.ndarray) -> np.ndarray:
        h = relu(obs.dot(self.W1) + self.b1)
        logits = h.dot(self.W2) + self.b2
        probs = softmax(logits)
        return probs, h

    def _value_forward(self, obs: np.ndarray) -> Tuple[float, np.ndarray]:
        h = relu(obs.dot(self.Wv1) + self.bv1)
        v = float(h.dot(self.Wv2).astype(float) + self.bv2)
        return v, h

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        obs = np.asarray(obs, dtype=float)
        probs, _ = self._policy_forward(obs)
        return int(np.argmax(probs)) if deterministic else int(np.random.choice(self.act_dim, p=probs))

    def store_transition(self, obs, action, reward):
        self._obs_buf.append(np.asarray(obs, dtype=float).copy())
        self._act_buf.append(int(action))
        self._rew_buf.append(float(reward))

    def _discounted_returns(self, rewards):
        R = np.zeros_like(rewards, dtype=float)
        running = 0.0
        for i in reversed(range(len(rewards))):
            running = rewards[i] + self.gamma * running
            R[i] = running
        if len(R) > 1:
            R = (R - R.mean()) / (R.std() + 1e-8)
        return R

    def update(self):
        if len(self._rew_buf) == 0:
            return
        obs = np.vstack(self._obs_buf)
        acts = np.asarray(self._act_buf, dtype=int)
        rets = self._discounted_returns(self._rew_buf)
        T = len(rets)

        h = relu(obs.dot(self.W1) + self.b1)
        logits = h.dot(self.W2) + self.b2
        probs = softmax(logits)

        if self.use_baseline:
            hv = relu(obs.dot(self.Wv1) + self.bv1)
            vals = (hv.dot(self.Wv2) + self.bv2).reshape(-1)
            adv = rets - vals
        else:
            adv = rets

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(T), acts] = 1.0
        grad_logits = (one_hot - probs) * adv.reshape(-1, 1)
        grad_W2 = h.T.dot(grad_logits) / float(T)
        grad_b2 = np.mean(grad_logits, axis=0)
        dh = grad_logits.dot(self.W2.T) * (h > 0).astype(float)
        grad_W1 = obs.T.dot(dh) / float(T)
        grad_b1 = np.mean(dh, axis=0)

        self.W2 += self.lr * grad_W2
        self.b2 += self.lr * grad_b2
        self.W1 += self.lr * grad_W1
        self.b1 += self.lr * grad_b1

        if self.use_baseline:
            err = (vals - rets).reshape(-1, 1)
            grad_v2 = hv.T.dot(err) / float(T)
            grad_bv2 = np.mean(err, axis=0)
            dh_v = err.dot(self.Wv2.T) * (hv > 0).astype(float)
            grad_v1 = obs.T.dot(dh_v) / float(T)
            grad_bv1 = np.mean(dh_v, axis=0)

            self.Wv2 -= self.lr * grad_v2
            self.bv2 -= self.lr * grad_bv2
            self.Wv1 -= self.lr * grad_v1
            self.bv1 -= self.lr * grad_bv1

        self._obs_buf.clear()
        self._act_buf.clear()
        self._rew_buf.clear()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(
            d["obs_dim"],
            d["act_dim"],
            hidden=d["hidden"],
            lr=d["lr"],
            gamma=d["gamma"],
            seed=None,
            use_baseline=d["use_baseline"],
        )
        obj.__dict__.update(d)
        return obj
