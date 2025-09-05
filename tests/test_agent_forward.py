# tests/test_agent_forward.py
import numpy as np
from ai.models.rl_agent import MLPAgent

def test_agent_select_update():
    obs_dim = 6
    act_dim = 3
    agent = MLPAgent(obs_dim=obs_dim, act_dim=act_dim, hidden=16, lr=1e-2, gamma=0.99, seed=123)
    # create synthetic episode
    for _ in range(10):
        obs = np.random.RandomState(0).randn(obs_dim)
        a = agent.select_action(obs, deterministic=False)
        agent.store_transition(obs, a, reward=1.0)
    # ensure buffers non-empty
    agent.update()
    # after update buffers cleared
    assert len(agent._obs_buf) == 0
    # test deterministic selection works
    obs2 = np.zeros(obs_dim)
    a_det = agent.select_action(obs2, deterministic=True)
    assert isinstance(a_det, int)
