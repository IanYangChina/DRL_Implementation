import random as R
from numpy import array_equal


class ReplayBuffer(object):
    def __init__(self, capacity, tr_namedtuple, seed=0):
        R.seed(seed)
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = tr_namedtuple

    def store_experience(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = R.sample(self.memory, batch_size)
        return self.Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)


class EpisodeWiseReplayBuffer(object):
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


"""
Below are two replay buffers with hindsight goal relabeling support.
    The first one is for general usage with transition tuple: 
        ('state', 'desired_goal', 'action', 'next_state', 'achieved_goal', 'reward', 'done').
    The second one is for the GridWorld environment designed in this repo ./envs/grid_world, with tuple:
        ('state', 'inventory', 'desired_goal', 'action', 
        'next_state', 'next_inventory', 'next_goal', 'achieved_goal', 
        'reward', 'done')
"""


class HindsightReplayBuffer(EpisodeWiseReplayBuffer):
    def __init__(self, capacity, tr_namedtuple, sampled_goal_num=6, seed=0):
        self.k = sampled_goal_num
        EpisodeWiseReplayBuffer.__init__(self, capacity, tr_namedtuple, seed)

    def modify_episodes(self):
        if len(self.episodes) == 0:
            return
        for _ in range(len(self.episodes)):
            ep = self.episodes[_]
            imagined_goals = self.sample_achieved_goal(ep)
            for n in range(len(imagined_goals[0])):
                ind = imagined_goals[0][n]
                goal = imagined_goals[1][n]
                modified_ep = []
                for tr in range(ind+1):
                    s = ep[tr].state
                    dg = goal
                    a = ep[tr].action
                    ns = ep[tr].next_state
                    ag = ep[tr].achieved_goal
                    r = ep[tr].reward
                    d = ep[tr].done
                    if tr == ind:
                        modified_ep.append(self.Transition(s, dg, a, ns, ag, -0.0, 0))
                    else:
                        modified_ep.append(self.Transition(s, dg, a, ns, ag, r, d))
                self.episodes.append(modified_ep)

    def sample_achieved_goal(self, ep):
        goals = [[], []]
        for _ in range(self.k):
            done = False
            count = 0
            while not done:
                count += 1
                if count > len(ep):
                    break
                ind = R.randint(0, len(ep)-1)
                goal = ep[ind].achieved_goal
                if all(not array_equal(goal, g) for g in goals[1]):
                    goals[0].append(ind)
                    goals[1].append(goal)
                    done = True
        return goals


class GridWorldHindsightReplayBuffer(EpisodeWiseReplayBuffer):
    def __init__(self, capacity, tr_namedtuple, sampled_goal_num=6, seed=0):
        self.k = sampled_goal_num
        EpisodeWiseReplayBuffer.__init__(self, capacity, tr_namedtuple, seed)

    def modify_episodes(self):
        if len(self.episodes) == 0:
            return
        for _ in range(len(self.episodes)):
            ep = self.episodes[_]
            imagined_goals = self.sample_achieved_goal(ep)
            for n in range(len(imagined_goals[0])):
                ind = imagined_goals[0][n]
                goal = imagined_goals[1][n]
                modified_ep = []
                for tr in range(ind+1):
                    s = ep[tr].state
                    inv = ep[tr].inventory
                    dg = goal
                    a = ep[tr].action
                    ns = ep[tr].next_state
                    ninv = ep[tr].next_inventory
                    ng = goal
                    ag = ep[tr].achieved_goal
                    r = ep[tr].reward
                    d = ep[tr].done
                    if tr == ind:
                        modified_ep.append(self.Transition(s, inv, dg, a, ns, ninv, ng, ag, 1.0, 0))
                    else:
                        modified_ep.append(self.Transition(s, inv, dg, a, ns, ninv, ng, ag, r, d))
                self.episodes.append(modified_ep)

    def sample_achieved_goal(self, ep):
        goals = [[], []]
        for k_ in range(self.k):
            done = False
            count = 0
            while not done:
                count += 1
                if count > len(ep):
                    break
                ind = R.randint(0, len(ep)-1)
                goal = ep[ind].achieved_goal
                if all(not array_equal(goal, g) for g in goals[1]):
                    goals[0].append(ind)
                    goals[1].append(goal)
                    done = True
        return goals