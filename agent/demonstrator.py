import random as R


class Demonstrator(object):
    def __init__(self, demonstrations, seed=0):
        R.seed(seed)
        self.demonstrations = demonstrations
        self.demon_num = len(self.demonstrations)
        self.demon_ind = 0
        self.current_goal = -1
        self.current_final_goal = 0

    def get_final_goal(self):
        return self.current_final_goal

    def get_next_goal(self):
        self.current_goal = (self.current_goal+1) % len(self.demonstrations[self.demon_ind])
        return self.demonstrations[self.demon_ind][self.current_goal]

    def reset(self):
        self.current_goal = -1
        self.demon_ind = R.randint(0, self.demon_num-1)
        self.current_final_goal = self.demonstrations[self.demon_ind][-1]

    def manual_reset(self, demon_ind):
        self.current_goal = -1
        self.demon_ind = demon_ind
        self.current_final_goal = self.demonstrations[self.demon_ind][-1]