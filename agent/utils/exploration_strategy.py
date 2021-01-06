import math as M
import numpy as np
from copy import deepcopy as dcp


class GoalSucRateBasedExpGreed(object):
    """
    This exploration class is a goal-success-rate-based exponentially-decaying epsilon-greedy exploration strategy.
    It modifies the original exponentially-decaying epsilon-greedy exploration strategy by reducing random exploration
        w.r.t. 1) the overall success rate of goals, and
               2) the success rate of each goal.
    """

    def __init__(self, goals, alpha=0.2, threshold=0.02, decay=None, sub_suc_percentage=None):
        self.alpha = alpha
        self.threshold = threshold
        if decay is None:
            self.decay = 50000
        else:
            self.decay = decay
        if sub_suc_percentage is None:
            self.sub_suc_percentage = 0.7
            self.avg_suc_percentage = 1 - self.sub_suc_percentage
        elif sub_suc_percentage > 1.0:
            raise ValueError("Sub-goal success rate percentage should be smaller than 1.0")
        else:
            self.sub_suc_percentage = sub_suc_percentage
            self.avg_suc_percentage = 1 - sub_suc_percentage
        self.goals = goals
        self.test_success_rates = np.zeros(len(self.goals))
        self.overall_success_rates = 0.0
        self.epsilons_start = np.ones(len(self.goals)) + self.threshold
        self.epsilons_epo = np.ones(len(self.goals)) + self.threshold

    def update_epsilons(self, new_tet_suc_rate):
        old_tet_suc_rate = dcp(self.test_success_rates)
        self.test_success_rates = (1 - self.alpha) * old_tet_suc_rate + self.alpha * new_tet_suc_rate
        self.overall_success_rates = np.mean(self.test_success_rates)
        self.epsilons_start = 1 - self.sub_suc_percentage * self.test_success_rates - self.avg_suc_percentage * self.overall_success_rates + self.threshold

    def print_epsilons(self, ep):
        self.epsilons_epo = self.threshold + (self.epsilons_start - self.threshold) * M.exp(-1. * ep / self.decay)
        return print("Current epsilons", self.epsilons_epo)

    def __call__(self, goal, ep):
        ind = self.goals.index(goal)
        return self.threshold + (self.epsilons_start[ind] - self.threshold) * M.exp(-1. * ep / self.decay)


class ExpDecayGreedy(object):
    def __init__(self, start=1, end=0.05, decay=50000, decay_start=None):
        self.start = start
        self.end = end
        self.decay = decay
        self.decay_start = decay_start

    def __call__(self, count):
        if self.decay_start is not None:
            count -= self.decay_start
            if count < 0:
                count = 0
        return self.end + (self.start - self.end) * M.exp(-1. * count / self.decay)


class LinearDecayGreedy(object):
    def __init__(self, start=1.0, end=0.1, decay=1000000, decay_start=None):
        self.start = start
        self.end = end
        self.decay = decay
        self.decay_start = decay_start

    def __call__(self, count):
        if self.decay_start is not None:
            count -= self.decay_start
            if count < 0:
                count = 0
            if count > self.decay:
                count = self.dacay
        return self.start - count * (self.start - self.end) / self.decay


class ConstantChance(object):
    def __init__(self, chance=0.2, rng=None):
        self.chance = chance
        if rng is None:
            self.rng = np.random.default_rng(seed=0)
        else:
            self.rng = rng

    def __call__(self):
        chance = self.rng.uniform(0, 1)
        if chance >= self.chance:
            return True
        else:
            return False


class OUNoise(object):
    # https://github.com/rll/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2, rng=None):
        if rng is None:
            self.rng = np.random.default_rng(seed=0)
        else:
            self.rng = rng
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * self.rng.standard_normal(len(x))
        self.state = x + dx
        return self.state
