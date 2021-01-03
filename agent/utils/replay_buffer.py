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
                if all(not np.array_equal(goal, g) for g in goals[1]):
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
                if all(not np.array_equal(goal, g) for g in goals[1]):
                    goals[0].append(ind)
                    goals[1].append(goal)
                    done = True
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
                 sampled_goal_num=4, reward_value=0.0, goal_type='state',
                 rng=None):
        self.k = sampled_goal_num
        self.r = reward_value
        self.goal_type = goal_type
        assert self.goal_type in ['state', 'image']
        PrioritisedEpisodeWiseReplayBuffer.__init__(self, capacity, tr_namedtuple, alpha=alpha, beta=beta, rng=rng)

    def modify_episodes(self):
        if len(self.episodes) == 0:
            return
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


def goal_distance_reward(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    d = np.linalg.norm(goal_a - goal_b, axis=-1)
    return -(d > 0.02).astype(np.float32)
