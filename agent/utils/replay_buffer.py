import random as R
import numpy as np
from .segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, capacity, tr_namedtuple, seed=0, saving_path=None):
        R.seed(seed)
        self.saving_path = saving_path
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

    def save_as_npy(self, start=None, end=None):
        assert self.saving_path is not None
        if start is None:
            batch = self.Transition(*zip(*self.memory))
        else:
            assert end is not None
            batch = self.Transition(*zip(*self.memory[start:end]))

        np.save(self.saving_path+'/observations', np.array(batch.observation))
        np.save(self.saving_path+'/actions', np.array(batch.action))
        np.save(self.saving_path+'/next_observations', np.array(batch.next_observation))
        np.save(self.saving_path+'/reward', np.array(batch.reward))
        np.save(self.saving_path+'/done', np.array(batch.done))

    def load_from_npy(self):
        assert self.saving_path is not None
        observations = np.load(self.saving_path+'/observations.npy')
        actions = np.load(self.saving_path+'/actions.npy')
        next_observations = np.load(self.saving_path+'/next_observations.npy')
        reward = np.load(self.saving_path+'/reward.npy')
        done = np.load(self.saving_path+'/done.npy')

        for i in range(observations.shape[0]):
            self.store_experience(observations[i],
                                  actions[i],
                                  next_observations[i],
                                  reward[i],
                                  done[i])

    def clear_memory(self):
        self.memory.clear()
        self.position = 0

    @property
    def full_memory(self):
        return self.Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


class EpisodeWiseReplayBuffer(object):
    def __init__(self, capacity, tr_namedtuple, seed=0):
        R.seed(seed)
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.new_episode = False
        self.episodes = []
        self.ep_position = -1
        self.Transition = tr_namedtuple

    def store_experience(self, *args):
        # $new_episode is a boolean value
        if self.new_episode:
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


class HindsightReplayBuffer(EpisodeWiseReplayBuffer):
    def __init__(self, capacity, tr_namedtuple, sampling_strategy='future', sampled_goal_num=6, seed=0):
        self.sampling_strategy = sampling_strategy
        assert self.sampling_strategy in ['final', 'episode', 'future']
        self.k = sampled_goal_num
        EpisodeWiseReplayBuffer.__init__(self, capacity, tr_namedtuple, seed)

    def modify_episodes(self):
        if len(self.episodes) == 0:
            return
        if self.sampling_strategy != 'future':
            # 'episode' or 'final' strategy
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
                        r = goal_distance_reward(dg, ag)
                        d = ep[tr].done
                        modified_ep.append(self.Transition(s, dg, a, ns, ag, r, d))
                    self.episodes.append(modified_ep)
        else:
            # 'future' strategy
            for _ in range(len(self.episodes)):
                ep = self.episodes[_]
                inds = R.sample(np.arange(len(ep)-1, dtype="int").tolist(), self.k)
                modified_ep = []
                for ind in inds:
                    s = ep[ind].state
                    dg = ep[ind+1].achieved_goal
                    a = ep[ind].action
                    ns = ep[ind].next_state
                    ag = ep[ind].achieved_goal
                    r = goal_distance_reward(dg, ag)
                    d = ep[ind].done
                    modified_ep.append(self.Transition(s, dg, a, ns, ag, r, d))
                self.episodes.append(modified_ep)

    def sample_achieved_goal(self, ep):
        goals = [[], []]
        if self.sampling_strategy == 'episode':
            for _ in range(self.k):
                done = False
                count = 0
                while not done:
                    count += 1
                    if count > len(ep):
                        break
                    ind = R.randint(0, len(ep)-1)
                    goal = ep[ind].achieved_goal
                    if all(not np.array_equal(goal, g) for g in goals[1]):
                        goals[0].append(ind)
                        goals[1].append(goal)
                        done = True
        elif self.sampling_strategy == 'final':
            goals[0].append(len(ep)-1)
            goals[1].append(ep[-1].achieved_goal)
        return goals


class PrioritisedReplayBuffer(object):
    def __init__(self, capacity, tr_namedtuple, alpha=0.5, beta=0.8, epsilon=1e-6, rng=None):
        if rng is None:
            self.rng = np.random.default_rng(seed=0)
        else:
            self.rng = rng
        self.capacity = capacity
        self.memory = []
        self.mem_position = 0
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

    def store_experience(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.mem_position] = self.Transition(*args)
        self.sum_tree[self.mem_position] = self._max_priority ** self.alpha
        self.min_tree[self.mem_position] = self._max_priority ** self.alpha
        self.mem_position = (self.mem_position + 1) % self.capacity

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

    def save_as_npy(self, start=None, end=None):
        assert self.saving_path is not None
        if start is None:
            batch = self.Transition(*zip(*self.memory))
        else:
            assert end is not None
            batch = self.Transition(*zip(*self.memory[start:end]))

        np.save(self.saving_path+'/observations', np.array(batch.observation))
        np.save(self.saving_path+'/actions', np.array(batch.action))
        np.save(self.saving_path+'/next_observations', np.array(batch.next_observation))
        np.save(self.saving_path+'/reward', np.array(batch.reward))
        np.save(self.saving_path+'/done', np.array(batch.done))

    def load_from_npy(self):
        assert self.saving_path is not None
        observations = np.load(self.saving_path+'/observations.npy')
        actions = np.load(self.saving_path+'/actions.npy')
        next_observations = np.load(self.saving_path+'/next_observations.npy')
        reward = np.load(self.saving_path+'/reward.npy')
        print(reward.sum())
        done = np.load(self.saving_path+'/done.npy')

        for i in range(observations.shape[0]):
            self.store_experience(observations[i],
                                  actions[i],
                                  next_observations[i],
                                  reward[i],
                                  done[i])

    def clear_memory(self):
        self.memory.clear()
        self.mem_position = 0

    @property
    def full_memory(self):
        return self.Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


class PrioritisedEpisodeWiseReplayBuffer(object):
    def __init__(self, capacity, tr_namedtuple, alpha=0.5, beta=0.8, epsilon=1e-6, rng=None):
        if rng is None:
            self.rng = np.random.default_rng(seed=0)
        else:
            self.rng = rng
        self.capacity = capacity
        self.memory = []
        self.mem_position = 0
        self.new_episode = False
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

    def store_experience(self, *args):
        if self.new_episode:
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
                 sampling_strategy='future', sampled_goal_num=4,
                 rng=None):
        self.sampling_strategy = sampling_strategy
        assert self.sampling_strategy in ['final', 'episode', 'future']
        self.k = sampled_goal_num
        PrioritisedEpisodeWiseReplayBuffer.__init__(self, capacity, tr_namedtuple, alpha=alpha, beta=beta, rng=rng)

    def modify_episodes(self):
        if len(self.episodes) == 0:
            return
        for _ in range(len(self.episodes)):
            if self.sampling_strategy != 'future':
                # 'episode' or 'final' strategy
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
                        r = goal_distance_reward(dg, ag)
                        d = ep[tr].done
                        modified_ep.append(self.Transition(s, dg, a, ns, ag, r, d))
                    self.episodes.append(modified_ep)
            else:
                # 'future' strategy
                ep = self.episodes[_]
                inds = R.sample(np.arange(len(ep)-1, dtype="int").tolist(), self.k)
                modified_ep = []
                for ind in inds:
                    s = ep[ind].state
                    dg = ep[ind+1].achieved_goal
                    a = ep[ind].action
                    ns = ep[ind].next_state
                    ag = ep[ind].achieved_goal
                    r = goal_distance_reward(dg, ag)
                    d = ep[ind].done
                    modified_ep.append(self.Transition(s, dg, a, ns, ag, r, d))
                self.episodes.append(modified_ep)

    def sample_achieved_goal(self, ep):
        goals = [[], []]
        if self.sampling_strategy == 'episode':
            for _ in range(self.k):
                done = False
                count = 0
                while not done:
                    count += 1
                    if count > len(ep):
                        break
                    ind = R.randint(0, len(ep)-1)
                    goal = ep[ind].achieved_goal
                    if all(not np.array_equal(goal, g) for g in goals[1]):
                        goals[0].append(ind)
                        goals[1].append(goal)
                        done = True
        elif self.sampling_strategy == 'final':
            goals[0].append(len(ep)-1)
            goals[1].append(ep[-1].achieved_goal)
        return goals


def goal_distance_reward(goal_a, goal_b):
    # sparse distance-based reward function for goal-conditioned env
    assert goal_a.shape == goal_b.shape
    d = np.linalg.norm(goal_a - goal_b, axis=-1)
    return -(d > 0.02).astype(np.float32)
