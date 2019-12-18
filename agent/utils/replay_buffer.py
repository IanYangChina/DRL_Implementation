import random as R


class ReplayBuffer(object):
    def __init__(self, capacity, tr_namedtuple, seed=0):
        R.seed(seed)
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.episodes = []
        self.ep_position = -1
        self.Transition = tr_namedtuple

    def store_experience(self, new_episode=False, *args):
        # $new_episode is a boolean value
        if new_episode:
            self.episodes.append([])
            self.ep_position += 1
        self.episodes[self.ep_position].append(self.Transition(*args))

    def store_episodes(self):
        if len(self.episodes) == 0:
            return
        for ep in self.episodes:
            for n in range(len(ep)):
                if len(self.memory) < self.capacity:
                    self.memory.append(None)
                self.memory[self.position] = ep[n]
                self.position = (self.position + 1) % self.capacity
        self.episodes.clear()
        self.ep_position = -1

    def sample(self, batch_size):
        batch = R.sample(self.memory, batch_size)
        return self.Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)