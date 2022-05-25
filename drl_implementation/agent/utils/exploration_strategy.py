import math as M
import numpy as np


class ExpDecayGreedy(object):
    # e-greedy exploration with exponential decay
    def __init__(self, start=1, end=0.05, decay=50000, decay_start=None, rng=None):
        self.start = start
        self.end = end
        self.decay = decay
        self.decay_start = decay_start
        if rng is None:
            self.rng = np.random.default_rng(seed=0)
        else:
            self.rng = rng

    def __call__(self, count):
        if self.decay_start is not None:
            count -= self.decay_start
            if count < 0:
                count = 0
        epsilon = self.end + (self.start - self.end) * M.exp(-1. * count / self.decay)
        prob = self.rng.uniform(0, 1)
        if prob < epsilon:
            return True
        else:
            return False


class LinearDecayGreedy(object):
    # e-greedy exploration with linear decay
    def __init__(self, start=1.0, end=0.1, decay=1000000, decay_start=None, rng=None):
        self.start = start
        self.end = end
        self.decay = decay
        self.decay_start = decay_start
        if rng is None:
            self.rng = np.random.default_rng(seed=0)
        else:
            self.rng = rng

    def __call__(self, count):
        if self.decay_start is not None:
            count -= self.decay_start
            if count < 0:
                count = 0
        if count > self.decay:
            count = self.decay
        epsilon = self.start - count * (self.start - self.end) / self.decay
        prob = self.rng.uniform(0, 1)
        if prob < epsilon:
            return True
        else:
            return False


class OUNoise(object):
    # https://github.com/rll/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
    def __init__(self, action_dim, action_max, mu=0, theta=0.2, sigma=1.0, rng=None):
        if rng is None:
            self.rng = np.random.default_rng(seed=0)
        else:
            self.rng = rng
        self.action_dim = action_dim
        self.action_max = action_max
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def __call__(self, action):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * self.rng.standard_normal(len(x))
        self.state = x + dx
        return np.clip(action + self.state, -self.action_max, self.action_max)


class GaussianNoise(object):
    # the one used in the TD3 paper: http://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf
    def __init__(self, action_dim, action_max, scale=1, mu=0, sigma=0.1, rng=None):
        if rng is None:
            self.rng = np.random.default_rng(seed=0)
        else:
            self.rng = rng
        self.scale = scale
        self.action_dim = action_dim
        self.action_max = action_max
        self.mu = mu
        self.sigma = sigma

    def __call__(self, action):
        noise = self.scale*self.rng.normal(loc=self.mu, scale=self.sigma, size=(self.action_dim,))
        return np.clip(action + noise, -self.action_max, self.action_max)


class EGreedyGaussian(object):
    # the one used in the HER paper: https://arxiv.org/abs/1707.01495
    def __init__(self, action_dim, action_max, chance=0.2, scale=1, mu=0, sigma=0.1, rng=None):
        self.chance = chance
        self.scale = scale
        self.action_dim = action_dim
        self.action_max = action_max
        self.mu = mu
        self.sigma = sigma
        if rng is None:
            self.rng = np.random.default_rng(seed=0)
        else:
            self.rng = rng

    def __call__(self, action):
        chance = self.rng.uniform(0, 1)
        if chance < self.chance:
            return self.rng.uniform(-self.action_max, self.action_max, size=(self.action_dim,))
        else:
            noise = self.scale*self.rng.normal(loc=self.mu, scale=self.sigma, size=(self.action_dim,))
            return np.clip(action + noise, -self.action_max, self.action_max)


class AutoAdjustingEGreedyGaussian(object):
    """
    https://ieeexplore.ieee.org/document/9366328
    This exploration class is a goal-success-rate-based auto-adjusting exploration strategy.
    It modifies the original constant chance exploration strategy by reducing exploration probabilities and noise deviations
        w.r.t. the testing success rate of each goal.
    """
    def __init__(self, goal_num, action_dim, action_max, tau=0.05, chance=0.2, scale=1, mu=0, sigma=0.2, rng=None):
        if rng is None:
            self.rng = np.random.default_rng(seed=0)
        else:
            self.rng = rng
        self.scale = scale
        self.action_dim = action_dim
        self.action_max = action_max
        self.mu = mu
        self.base_sigma = sigma
        self.sigma = np.ones(goal_num) * sigma

        self.base_chance = chance
        self.goal_num = goal_num
        self.tau = tau
        self.success_rates = np.zeros(self.goal_num)
        self.chance = np.ones(self.goal_num) * chance

    def update_success_rates(self, new_tet_suc_rate):
        old_tet_suc_rate = self.success_rates.copy()
        self.success_rates = (1-self.tau)*old_tet_suc_rate + self.tau*new_tet_suc_rate
        self.chance = self.base_chance*(1-self.success_rates)
        self.sigma = self.base_sigma*(1-self.success_rates)

    def __call__(self, goal_ind, action):
        # return a random action or a noisy action
        prob = self.rng.uniform(0, 1)
        if prob < self.chance[goal_ind]:
            return self.rng.uniform(-self.action_max, self.action_max, size=(self.action_dim,))
        else:
            noise = self.scale*self.rng.normal(loc=self.mu, scale=self.sigma[goal_ind], size=(self.action_dim,))
            return action + noise
