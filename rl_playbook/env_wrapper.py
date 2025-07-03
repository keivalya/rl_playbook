import gymnasium as gym

class EnvWrapper:
    def __init__(self, env, normalize=False, reward_scale=1.0):
        self.env = gym.make(env)
        self.normalize = normalize
        self.reward_scale = reward_scale

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        if action is None:
            raise ValueError("Action cannot be None")
        next_state, reward, terminated, truncated, info = self.env.step(action)
        if self.normalize:
            next_state = (next_state - next_state.mean()) / (next_state.std() + 1e-8)
        reward = reward * self.reward_scale
        return next_state, reward, terminated, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()
    
    def seed(self, seed=None):
        return self.env.seed(seed)
    
    def action_space(self):
        return self.env.action_space
    
    def observation_space(self):
        return self.env.observation_space
    
    def state(self):
        return self.env.state
    
    def reward(self):
        return self.env.reward
    
    def done(self):
        return self.env.done
    
    def info(self):
        return self.env.info
    
    def is_terminated(self):
        return self.env.is_terminated
    
    def is_truncated(self):
        return self.env.is_truncated