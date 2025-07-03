# This file is the main file for the RLPlaybook package.
# Example usage:
# from rl_playbook import TrainLoop, EnvWrapper, Evaluator
# loop = TrainLoop(
#     env=EnvWrapper("CartPole-v1", normalize=True, reward_scale=0.1),
#     agent="PPO",
#     total_timesteps=100_000,
#     evaluator=Evaluator(eval_env="CartPole-v1", every=5_000),
#     log_to="tensorboard"
# )
# loop.run()

from env_wrapper import EnvWrapper
from agent import Agent
from evaluator import Evaluator

class RLPlaybook:
    def __init__(self, env, agent, total_timesteps, evaluator, log_to):
        self.env = env
        self.agent = agent
        self.total_timesteps = total_timesteps
        self.evaluator = evaluator
        self.log_to = log_to
    
    def TrainLoop(self, env: EnvWrapper, agent: Agent, total_timesteps: int, evaluator: Evaluator, log_to: str):
        """
        Train the agent for the given number of timesteps.
        """
        for episode in range(total_timesteps):
            state = env.reset()
            terminated = False
            while not terminated:
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.learn(state, action, reward, next_state, terminated, truncated)
                state = next_state
                if terminated or truncated:
                    break
            if evaluator is not None:
                evaluator.evaluate()
            if log_to is not None:
                pass

    def run(self):
        self.TrainLoop(self.env, self.agent, self.total_timesteps, self.evaluator, self.log_to)

    def train(self):
        self.TrainLoop(self.env, self.agent, self.total_timesteps, self.evaluator, self.log_to)

    def evaluate(self):
        self.evaluator.evaluate(self.env, self.agent, self.total_timesteps, self.evaluator, self.log_to)

    def log(self):
        self.log_to.log(self.env, self.agent, self.total_timesteps, self.evaluator, self.log_to)
    
    