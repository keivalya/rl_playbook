import unittest
import numpy as np
import random

import gymnasium as gym
from q_learning import QLearning

class TestQLearning(unittest.TestCase):
    def test_q_learning(self):
        env = gym.make("FrozenLake-v1")
        q_learning = QLearning(env.observation_space.n, env.action_space.n, 0.1, 0.9, 0.1, 100, env)
        q_learning.run()
        self.assertEqual(q_learning.Q.shape, (env.observation_space.n, env.action_space.n))
        self.assertEqual(q_learning.Q.sum(), 0)
        self.assertEqual(q_learning.Q.sum(), 0)

if __name__ == "__main__":
    unittest.main()