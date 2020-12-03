import random as R
from numpy import array_equal
from .segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, capacity, tr_namedtuple, seed=0):
        R.seed(seed)
        self.capacity = capacity
        self.memory = []
        self.position = 0  # 99, rewrite from the 0-th transition
        self.Transition = tr_namedtuple

    def store_experience(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = R.sample(self.memory, batch_size)  # uniform sampling
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


class PrioritisedEpisodeWiseReplayBuffer(object):
    def __init__(self, capacity, tr_namedtuple, alpha=0.5, beta=0.8, epsilon=1e-6, rng=None):
        if rng is None:
            self.rng = np.random.default_rng(seed=0)
        else:
            self.rng = rng
        self.capacity = capacity
        self.memory = []
        self.mem_position = 0
        self.episodes = []
        self.ep_position = -1
        self.Transition = tr_namedtuple
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self._max_priority = 1.0

    def store_experience(self, new_episode=False, *args):
        """When new_episode is True, create a new list to store data.
        This can be used to separate data storage for purposes such as trajectory-wise relabelling.
        """
        if new_episode:
            self.episodes.append([])
            self.ep_position += 1
        self.episodes[self.ep_position].append(self.Transition(*args))

    def store_episode(self):
        if len(self.episodes) == 0:
            return
        for ep in self.episodes:
            for n in range(len(ep)):
                if len(self.memory) < self.capacity:
                    self.memory.append(None)
                self.memory[self.mem_position] = ep[n]
                self.sum_tree[self.mem_position] = self._max_priority ** self.alpha
                self.min_tree[self.mem_position] = self._max_priority ** self.alpha
                self.mem_position = (self.mem_position + 1) % self.capacity
        self.episodes.clear()
        self.ep_position = -1

    def sample(self, batch_size, beta=None):
        if beta is None:
            beta = self.beta
        assert beta > 0, "beta should be greater than 0"
        inds, priority_sum = self.sample_proportion(batch_size)
        batch = []
        weights = []
        minimal_priority = self.min_tree.min()
        max_weight = (minimal_priority / priority_sum * len(self)) ** (-beta)
        for ind in inds:
            batch.append(self.memory[ind])
            sample_priority = self.sum_tree[ind] / priority_sum
            weight = (sample_priority * len(self)) ** (-beta)
            weight = weight / max_weight
            weights.append(weight)

        return self.Transition(*zip(*batch)), np.array(weights), inds

    def sample_proportion(self, batch_size):
        inds = []
        priority_sum = self.sum_tree.sum(0, len(self) - 1)
        interval = priority_sum / batch_size
        for i in range(batch_size):
            mass = self.rng.uniform() * interval + i * interval
            ind = self.sum_tree.find_prefixsum_idx(mass)
            try:
                k = self.memory[ind]
            except IndexError as e:
                print(e, ind, len(self))

            inds.append(ind)
        return inds, priority_sum

    def update_priority(self, inds, priorities):
        for ind, priority in zip(inds, priorities):
            assert priority >= 0
            assert 0 <= ind < len(self)
            self.sum_tree[ind] = (priority + self.epsilon) ** self.alpha
            self.min_tree[ind] = (priority + self.epsilon) ** self.alpha
            self._max_priority = max(self._max_priority, priority)

    def __len__(self):
        return len(self.memory)


class PrioritisedHindsightReplayBuffer(PrioritisedEpisodeWiseReplayBuffer):
    def __init__(self, capacity, tr_namedtuple, alpha=0.5, beta=0.8,
                 sampled_goal_num=4, reward_value=0.0, level='low', goal_type='state',
                 rng=None):
        self.k = sampled_goal_num
        self.r = reward_value
        assert level in ['low', 'high']
        self.lv = level
        self.goal_type = goal_type
        assert self.goal_type in ['state', 'image']
        PrioritisedEpisodeWiseReplayBuffer.__init__(self, capacity, tr_namedtuple, alpha=alpha, beta=beta, rng=rng)

    def modify_experiences(self):
        if len(self.episodes) == 0:
            return
        if self.lv == 'low':
            for _ in range(len(self.episodes)):
                ep = self.episodes[_]
                if len(ep) < 2:
                    continue
                goals = self.sample_achieved_goal_random(ep)
                for n in range(len(goals[0])):
                    # ind = goals[0][n]
                    # goal = goals[1][n]
                    modified_ep = []
                    for tr in range(goals[0][n]+1):
                        s = ep[tr].state
                        dg = goals[1][n]
                        a = ep[tr].action
                        ns = ep[tr].next_state
                        ag = ep[tr].achieved_goal
                        r = goal_distance_reward(dg, ag)
                        d = ep[tr].done
                        if self.goal_type == 'image':
                            obs = ep[tr].observation
                            dgi = goals[2][n]
                            nobs = ep[tr].next_observation
                            agi = ep[tr].achieved_goal_image
                            if tr == goals[0][n]:
                                modified_ep.append(self.Transition(s, obs, dg, dgi, a, ns, nobs, ag, agi, r, 0))
                            else:
                                modified_ep.append(self.Transition(s, obs, dg, dgi, a, ns, nobs, ag, agi, r, d))
                        else:
                            if tr == goals[0][n]:
                                modified_ep.append(self.Transition(s, dg, a, ns, ag, r, 0))
                            else:
                                modified_ep.append(self.Transition(s, dg, a, ns, ag, r, d))
                    self.episodes.append(modified_ep)
        else:
            for _ in range(len(self.episodes)):
                ep = self.episodes[_]
                if len(ep) < 2:
                    continue
                goals = self.sample_achieved_goal_random(ep)
                for n in range(len(goals[0])):
                    ind = goals[0][n]
                    goal = goals[1][n]
                    modified_ep = []
                    o_done_count = 0
                    for tr in range(1, ind + 1):
                        # index starts from -1 towards 0
                        s = ep[-tr].state
                        dg = goal
                        if o_done_count == 0:
                            o = goal
                        else:
                            o = ep[-tr].option
                        ns = ep[-tr].next_state
                        ag = ep[-tr].achieved_goal
                        o_done = ep[-tr].option_done
                        if o_done == 0:
                            o_done_count += 1
                        r = ep[-tr].reward
                        d = ep[-tr].done
                        if tr == 1:
                            modified_ep.append(self.Transition(s, dg, o, ns, ag, 0, self.r, 0))
                        else:
                            modified_ep.append(self.Transition(s, dg, o, ns, ag, o_done, r, d))
                    self.episodes.append(modified_ep)

    def sample_achieved_goal_random(self, ep):
        assert (len(ep)-1) >= 0
        goals = [[], [], []]
        for k_ in range(self.k):
            done = False
            while not done:
                ind = self.rng.integers(5, len(ep) - 1)
                goal = ep[ind].achieved_goal
                if all(not np.array_equal(goal, g) for g in goals[1]):
                    if self.goal_type == 'image':
                        goals[2].append(ep[ind].achieved_goal_image)
                    goals[1].append(goal)
                    goals[0].append(ind)
                    done = True
        return goals