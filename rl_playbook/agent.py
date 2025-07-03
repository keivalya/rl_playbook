class Agent:
    def __init__(self, agent_type):
        self.agent_type = agent_type

    def train(self):
        pass

    def act(self, state):
        pass

    def learn(self, state, action, reward, next_state, terminated, truncated):
        pass