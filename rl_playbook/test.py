import unittest
import numpy as np
import random

from rl_playbook import RLPlaybook
from env_wrapper import EnvWrapper
from agent import Agent
from evaluator import Evaluator

class TestRLPlaybook(unittest.TestCase):
    def test_rl_playbook(self):
        # Create a mock agent that returns valid actions
        class MockAgent(Agent):
            def act(self, state):
                # Return a random valid action for CartPole (0 or 1)
                return random.randint(0, 1)
            
            def learn(self, state, action, reward, next_state, terminated, truncated):
                pass

        env = EnvWrapper("CartPole-v1")
        agent = MockAgent("PPO")  # Use mock agent instead
        evaluator = Evaluator("CartPole-v1", 10)
        log_to = "tensorboard"
        rl_playbook = RLPlaybook(env, agent, 100, evaluator, log_to)
        rl_playbook.run()

if __name__ == "__main__":
    unittest.main()