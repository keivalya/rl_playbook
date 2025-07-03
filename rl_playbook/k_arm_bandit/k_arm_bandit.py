import random
import numpy as np

class BernoulliBandit:
    def __init__(self, p: int, verbose: bool = False):
        """
        p: probability of success
        """
        self.p = p
        if verbose:
            print("Creating BernoulliBandit with p = {:.2f}".format(p))

    def pull(self):
        """
        Pull the arm and return the reward
        """
        return np.random.binomial(1, self.p)
    
class BanditGame:
    def __init__(self, k: int, n: int, verbose: bool = False):
        """
        k: number of arms
        n: number of trials
        """
        self.k = k
        self.n = n
        self.verbose = verbose
        self.bandits = [BernoulliBandit(np.random.uniform(), verbose) for i in range(k)]

    def play(self):
        """
        Play the game and return the results
        """
        results = np.zeros((self.n))
        for t in range(self.n):
            k = random.randrange(self.k)
            results[t] = self.bandits[k].pull()
            if self.verbose:
                print("T={} \t Playing bandit {} \t Reward is {:.2f}".format(t, k, results[t]))
        return results