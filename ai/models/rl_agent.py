"""
Reinforcement Learning Agent
"""

import random


class RLAgent:
    def __init__(self, actions=["BUY", "SELL", "HOLD"]):
        self.actions = actions

    def act(self, state):
        # TODO: replace with real policy (DQN, PPO, etc.)
        return random.choice(self.actions)

    def learn(self, state, action, reward, next_state):
        # TODO: implement RL training loop
        pass
