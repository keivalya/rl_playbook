from k_arm_bandit import *

# action-value function
# Q(a) = E[R | a]
# Q(a) = sum(R | a) / sum(1 | a)

# action-value function for each arm
# Q(a) = E[R | a]
# Q(a) = sum(R | a) / sum(1 | a)

# action-value function for each arm
class Simulator:
    def __init__(self, k: int, n: int, bandit: BanditGame, verbose: bool = False):
        self.k = k
        self.n = n
        self.verbose = verbose
        self.bandit = bandit
        self.Q = np.zeros(k)
        self.N = np.zeros(k)

    def play(self):
        pass

    def update(self, arm: int, reward: int):
        self.N[arm] += 1
        self.Q[arm] = self.Q[arm] + (1 / self.N[arm]) * (reward - self.Q[arm])

    def choose_arm(self):
        return np.argmax(self.Q)
    
    def run(self):
        for t in range(self.n):
            arm = self.choose_arm()
            reward = self.bandit.bandits[arm].pull()
            self.update(arm, reward)
            if self.verbose:
                print("T={} \t Playing bandit {} \t Reward is {:.2f}".format(t, arm, reward))
        return self.Q
    
    def run_greedy(self, epsilon: float = 0.1):
        for t in range(self.n):
            if np.random.random() < epsilon:
                arm = np.random.randint(self.k)
            else:
                arm = self.choose_arm()
            reward = self.bandit.bandits[arm].pull()
            self.update(arm, reward)
            if self.verbose:
                print("T={} \t Playing bandit {} \t Reward is {:.2f}".format(t, arm, reward))
        return self.Q