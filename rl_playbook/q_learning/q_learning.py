import numpy as np

# only for discrete action spaces!!

class QLearning:
    def __init__(self, n_states, n_actions, alpha, gamma, epsilon, n_episodes, env):
        """
        n_states: number of states
        n_actions: number of actions
        alpha: learning rate
        gamma: discount factor
        epsilon: exploration rate
        n_episodes: number of episodes
        env: environment
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_episodes = n_episodes
        self.Q = np.zeros((n_states, n_actions))
        self.env = env

    def choose_action(self, state):
        """
        Choose an action based on the current state
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
        
    def update_Q(self, state, action, reward, next_state):
        """
        Update the Q-value for the current state and action
        """
        self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action])
    
    def run(self):
        """
        Run the Q-learning algorithm
        """
        self.env.reset()
        for episode in range(self.n_episodes):
            state = np.random.randint(self.n_states)
            action = self.choose_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            self.update_Q(state, action, reward, next_state)
            state = next_state
            
    def run_greedy(self):
        """
        Run the Q-learning algorithm with greedy action selection
        """
        self.env.reset()
        for episode in range(self.n_episodes):
            state = np.random.randint(self.n_states)
            action = self.choose_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.update_Q(state, action, reward, next_state)
            state = next_state
    
    def run_epsilon_greedy(self):
        """
        Run the Q-learning algorithm with epsilon-greedy action selection
        """
        self.env.reset()
        for episode in range(self.n_episodes):
            state = np.random.randint(self.n_states)
            action = self.choose_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.update_Q(state, action, reward, next_state)
            state = next_state

    def run_softmax(self):
        """
        Run the Q-learning algorithm with softmax action selection
        """
        self.env.reset()
        for episode in range(self.n_episodes):
            state = np.random.randint(self.n_states)
            action = self.choose_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.update_Q(state, action, reward, next_state)
            state = next_state

    def run_softmax_epsilon_greedy(self):
        """
        Run the Q-learning algorithm with softmax and epsilon-greedy action selection
        """
        self.env.reset()
        for episode in range(self.n_episodes):
            state = np.random.randint(self.n_states)
            action = self.choose_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.update_Q(state, action, reward, next_state)
            state = next_state

    def run_ucb(self):
        """
        Run the Q-learning algorithm with UCB action selection
        """
        self.env.reset()
        for episode in range(self.n_episodes):
            state = np.random.randint(self.n_states)
            action = self.choose_action(state)
            reward = self.env.step(action)
            next_state = self.env.state
            self.update_Q(state, action, reward, next_state)
            state = next_state

    def run_ucb_epsilon_greedy(self):
        """
        Run the Q-learning algorithm with UCB and epsilon-greedy action selection
        """
        for episode in range(self.n_episodes):
            state = np.random.randint(self.n_states)
            action = self.choose_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.update_Q(state, action, reward, next_state)
            state = next_state

    def run_ts(self):
        """
        Run the Q-learning algorithm with Thompson Sampling action selection
        """
        for episode in range(self.n_episodes):
            state = np.random.randint(self.n_states)
            action = self.choose_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.update_Q(state, action, reward, next_state)
            state = next_state

    def run_ts_epsilon_greedy(self):
        """
        Run the Q-learning algorithm with Thompson Sampling and epsilon-greedy action selection
        """
        for episode in range(self.n_episodes):
            state = np.random.randint(self.n_states)
            action = self.choose_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.update_Q(state, action, reward, next_state)
            state = next_state  

    def run_dqn(self):
        """
        Run the Q-learning algorithm with DQN action selection
        """
        for episode in range(self.n_episodes):
            state = np.random.randint(self.n_states)
            action = self.choose_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.update_Q(state, action, reward, next_state)
            state = next_state

    def run_dqn_epsilon_greedy(self):
        """
        Run the Q-learning algorithm with DQN and epsilon-greedy action selection
        """
        for episode in range(self.n_episodes):
            state = np.random.randint(self.n_states)
            action = self.choose_action(state)
            reward = self.env.step(action)
            next_state = self.env.state
            self.update_Q(state, action, reward, next_state)
            state = next_state

    def run_dqn_softmax(self):
        """
        Run the Q-learning algorithm with DQN and softmax action selection
        """
        for episode in range(self.n_episodes):
            state = np.random.randint(self.n_states)
            action = self.choose_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.update_Q(state, action, reward, next_state)
            state = next_state

    def run_dqn_ucb(self):
        for episode in range(self.n_episodes):
            state = np.random.randint(self.n_states)
            action = self.choose_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.update_Q(state, action, reward, next_state)
            state = next_state

    def run_dqn_ts(self):
        """
        Run the Q-learning algorithm with DQN and Thompson Sampling action selection
        """
        for episode in range(self.n_episodes):
            state = np.random.randint(self.n_states)
            action = self.choose_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.update_Q(state, action, reward, next_state)
            state = next_state
