import math as M
import numpy as np
from copy import deepcopy as dcp


class GoalSucRateBased(object):
    def __init__(self, goals, alpha=0.2):
        self.alpha = alpha
        self.goals = goals
        self.success_rates = np.zeros(len(self.goals))

    def update_success_rate(self, new_suc_rate):
        if not isinstance(new_suc_rate, np.ndarray):
            raise TypeError("success rates must be of type ndarray, got {} instead".format(new_suc_rate.__class__.__name__))
        old_suc_rate = dcp(self.success_rates)
        self.success_rates = (1-self.alpha)*old_suc_rate + self.alpha*new_suc_rate

    def print_success_rates(self):
        return print("Current success rates", self.success_rates)

    def __call__(self, goal):
        ind = self.goals.index(goal)
        return 1 - self.success_rates[ind]


class DecayGreedy(object):
    def __init__(self,  start=1, end=0.05, decay=50000, decay_start=None):
        self.start = start
        self.end = end
        self.decay = decay
        self.decay_start = decay_start

    def __call__(self, count):
        if self.decay_start is None:
            epsilon = self.end + (self.start - self.end) * M.exp(-1. * count / self.decay)
        else:
            count -= self.decay_start
            if count < 0:
                count = 0
            epsilon = self.end + (self.start - self.end) * M.exp(-1. * count / self.decay)
        return epsilon


