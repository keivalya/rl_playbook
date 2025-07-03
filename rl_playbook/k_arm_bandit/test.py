import unittest
import numpy as np
import random

from k_arm_bandit import BernoulliBandit, BanditGame
from simulate import Simulator

class TestBernoulliBandit(unittest.TestCase):
    def test_pull_values_and_distribution(self):
        p = 0.8
        bandit = BernoulliBandit(p)
        pulls = [bandit.pull() for _ in range(10_000)]
        self.assertTrue(all(r in (0, 1) for r in pulls))
        emp_mean = np.mean(pulls)
        self.assertAlmostEqual(emp_mean, p, delta=0.05)

class TestBanditGame(unittest.TestCase):
    def test_init_and_play(self):
        k, n = 5, 200
        game = BanditGame(k, n)
        self.assertEqual(len(game.bandits), k)
        results = game.play()
        self.assertEqual(results.shape, (n,))
        self.assertTrue(all(r in (0, 1) for r in results))

class DummyBandit:
    def __init__(self, seq):
        self.seq = seq
        self.i = 0
    def pull(self):
        v = self.seq[self.i]
        self.i = (self.i + 1) % len(self.seq)
        return v

class DummyBanditGame:
    def __init__(self, bandits):
        self.bandits = bandits

class TestSimulator(unittest.TestCase):
    def test_update(self):
        sim = Simulator(k=2, n=1, bandit=DummyBanditGame([]))
        np.testing.assert_array_equal(sim.N, [0, 0])
        np.testing.assert_array_equal(sim.Q, [0.0, 0.0])

        sim.update(0, 1)
        self.assertEqual(sim.N[0], 1)
        self.assertAlmostEqual(sim.Q[0], 1.0)

        sim.update(0, 0)
        self.assertEqual(sim.N[0], 2)
        self.assertAlmostEqual(sim.Q[0], 0.5)

    def test_choose_arm(self):
        sim = Simulator(k=3, n=1, bandit=DummyBanditGame([]))
        sim.Q = np.array([0.2, 0.9, 0.5])
        self.assertEqual(sim.choose_arm(), 1)

    def test_run_converges(self):
        b0 = DummyBandit([1])
        b1 = DummyBandit([0])
        bandit_game = DummyBanditGame([b0, b1])
        sim = Simulator(k=2, n=50, bandit=bandit_game)
        Q = sim.run()
        self.assertAlmostEqual(Q[0], 1.0, places=3)
        self.assertAlmostEqual(Q[1], 0.0, places=3)

    def test_run_greedy_with_zero_epsilon(self):
        b0 = DummyBandit([1])
        b1 = DummyBandit([0])
        bandit_game = DummyBanditGame([b0, b1])

        sim1 = Simulator(k=2, n=50, bandit=bandit_game)
        Q1 = sim1.run()

        b0 = DummyBandit([1])
        b1 = DummyBandit([0])
        bandit_game = DummyBanditGame([b0, b1])

        sim2 = Simulator(k=2, n=50, bandit=bandit_game)
        Q2 = sim2.run_greedy(epsilon=0.0)

        np.testing.assert_allclose(Q1, Q2, atol=1e-6)

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    unittest.main()
