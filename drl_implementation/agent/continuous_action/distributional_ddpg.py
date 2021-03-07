import time
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from ..utils.networks_mlp import Actor, Critic
from ..agent_base import Agent
from ..utils.exploration_strategy import OUNoise, GaussianNoise


class DistributionalDDPG(Agent):
    def __init__(self, algo_params, env, transition_tuple=None, path=None, seed=-1):
        # environment
        self.env = env
        self.env.seed(seed)
        obs = self.env.reset()
        algo_params.update({'state_dim': obs.shape[0],
                            'action_dim': self.env.action_space.shape[0],
                            'action_max': self.env.action_space.high,
                            'action_scaling': self.env.action_space.high[0],
                            'init_input_means': None,
                            'init_input_vars': None
                            })
        # training args
        self.training_episodes = algo_params['training_episodes']
        self.testing_gap = algo_params['testing_gap']
        self.testing_episodes = algo_params['testing_episodes']
        self.saving_gap = algo_params['saving_gap']

        super(DistributionalDDPG, self).__init__(algo_params,
                                                 transition_tuple=transition_tuple,
                                                 goal_conditioned=False,
                                                 path=path,
                                                 seed=seed)
        # torch
        # categorical distribution atoms
        self.num_atoms = algo_params['num_atoms']
        self.value_max = algo_params['value_max']
        self.value_min = algo_params['value_min']
        self.delta_z = (self.value_max - self.value_min) / (self.num_atoms - 1)
        self.support = T.linspace(self.value_min, self.value_max, steps=self.num_atoms).to(self.device)
        # network
        self.network_dict.update({
            'actor': Actor(self.state_dim, self.action_dim).to(self.device),
            'actor_target': Actor(self.state_dim, self.action_dim).to(self.device),
            'critic': Critic(self.state_dim + self.action_dim, self.num_atoms, softmax=True).to(self.device),
            'critic_target': Critic(self.state_dim + self.action_dim, self.num_atoms, softmax=True).to(self.device)
        })
        self.network_keys_to_save = ['actor_target', 'critic_target']
        self.actor_optimizer = Adam(self.network_dict['actor'].parameters(), lr=self.actor_learning_rate)
        self._soft_update(self.network_dict['actor'], self.network_dict['actor_target'], tau=1)
        self.critic_optimizer = Adam(self.network_dict['critic'].parameters(), lr=self.critic_learning_rate,
                                     weight_decay=algo_params['Q_weight_decay'])
        self._soft_update(self.network_dict['critic'], self.network_dict['critic_target'], tau=1)
        # behavioural policy args (exploration)
        self.exploration_strategy = GaussianNoise(self.action_dim, scale=0.3, sigma=1.0)
        # training args
        self.warmup_step = algo_params['warmup_step']
        # statistic dict
        self.statistic_dict.update({
            'episode_return': [],
            'episode_test_return': []
        })

    def run(self, test=False, render=False, load_network_ep=None, sleep=0):
        if test:
            num_episode = self.testing_episodes
            if load_network_ep is not None:
                print("Loading network parameters...")
                self._load_network(ep=load_network_ep)
            print("Start testing...")
        else:
            num_episode = self.training_episodes
            print("Start training...")

        for ep in range(num_episode):
            ep_return = self._interact(render, test, sleep=sleep)
            self.statistic_dict['episode_return'].append(ep_return)
            print("Episode %i" % ep, "return %0.1f" % ep_return)

            if (ep % self.testing_gap == 0) and (ep != 0) and (not test):
                ep_test_return = []
                for test_ep in range(self.testing_episodes):
                    ep_test_return.append(self._interact(render, test=True))
                self.statistic_dict['episode_test_return'].append(sum(ep_test_return) / self.testing_episodes)
                print("Episode %i" % ep, "test return %0.1f" % (sum(ep_test_return) / self.testing_episodes))

            if (ep % self.saving_gap == 0) and (ep != 0) and (not test):
                self._save_network(ep=ep)

        if not test:
            print("Finished training")
            print("Saving statistics...")
            self._plot_statistics(save_to_file=True)
        else:
            print("Finished testing")

    def _interact(self, render=False, test=False, sleep=0):
        done = False
        obs = self.env.reset()
        ep_return = 0
        while not done:
            if render:
                self.env.render()
            if self.env_step_count < self.warmup_step:
                action = self.env.action_space.sample()
            else:
                action = self._select_action(obs, test=test)
            new_obs, reward, done, info = self.env.step(action)
            time.sleep(sleep)
            ep_return += reward
            if not test:
                self._remember(obs, action, new_obs, reward, 1 - int(done))
                if self.observation_normalization:
                    self.normalizer.store_history(new_obs)
                    self.normalizer.update_mean()
                if (self.env_step_count % self.update_interval == 0) and (self.env_step_count > self.warmup_step):
                    self._learn()
                self.env_step_count += 1
            obs = new_obs
        return ep_return

    def _select_action(self, obs, test=False):
        obs = self.normalizer(obs)
        with T.no_grad():
            inputs = T.as_tensor(obs, dtype=T.float, device=self.device)
            action = self.network_dict['actor_target'](inputs).cpu().detach().numpy()
        if test:
            # evaluate
            return np.clip(action, -self.action_max, self.action_max)
        else:
            # explore
            return self.exploration_strategy(action)

    def _learn(self, steps=None):
        if len(self.buffer) < self.batch_size:
            return
        if steps is None:
            steps = self.optimizer_steps

        for i in range(steps):
            if self.prioritised:
                batch, weights, inds = self.buffer.sample(self.batch_size)
                weights = T.as_tensor(weights, device=self.device).view(self.batch_size, 1)
            else:
                batch = self.buffer.sample(self.batch_size)
                weights = T.ones(size=(self.batch_size, 1), device=self.device)
                inds = None

            actor_inputs = self.normalizer(batch.state)
            actor_inputs = T.as_tensor(actor_inputs, dtype=T.float32, device=self.device)
            actions = T.as_tensor(batch.action, dtype=T.float32, device=self.device)
            critic_inputs = T.cat((actor_inputs, actions), dim=1)
            actor_inputs_ = self.normalizer(batch.next_state)
            actor_inputs_ = T.as_tensor(actor_inputs_, dtype=T.float32, device=self.device)
            rewards = T.as_tensor(batch.reward, dtype=T.float32, device=self.device)
            done = T.as_tensor(batch.done, dtype=T.float32, device=self.device)

            if self.discard_time_limit:
                done = done * 0 + 1

            with T.no_grad():
                actions_ = self.network_dict['actor_target'](actor_inputs_)
                critic_inputs_ = T.cat((actor_inputs_, actions_), dim=1)
                value_dist_ = self.network_dict['critic_target'](critic_inputs_)
                value_dist_target = self.project_value_distribution(value_dist_, rewards, done)
                value_dist_target = T.as_tensor(value_dist_target, device=self.device)

            self.critic_optimizer.zero_grad()
            value_dist_estimate = self.network_dict['critic'](critic_inputs)
            critic_loss = F.binary_cross_entropy(value_dist_estimate, value_dist_target, reduction='none').sum(dim=1)
            (critic_loss * weights).mean().backward()
            self.critic_optimizer.step()

            if self.prioritised:
                assert inds is not None
                self.buffer.update_priority(inds, np.abs(critic_loss.cpu().detach().numpy()))

            self.actor_optimizer.zero_grad()
            new_actions = self.network_dict['actor'](actor_inputs)
            critic_eval_inputs = T.cat((actor_inputs, new_actions), dim=1)
            # take the expectation of the value distribution as the policy loss
            actor_loss = -(self.network_dict['critic'](critic_eval_inputs) * self.support)
            actor_loss = actor_loss.sum(dim=1)
            actor_loss.mean().backward()
            self.actor_optimizer.step()

            self._soft_update(self.network_dict['actor'], self.network_dict['actor_target'])
            self._soft_update(self.network_dict['critic'], self.network_dict['critic_target'])

            self.statistic_dict['critic_loss'].append(critic_loss.detach().mean())
            self.statistic_dict['actor_loss'].append(actor_loss.detach().mean())

    def project_value_distribution(self, value_dist, rewards, done):
        # refer to https://github.com/schatty/d4pg-pytorch/blob/7dc23096a45bc4036fbb02493e0b052d57cfe4c6/models/d4pg/l2_projection.py#L7
        # comments added
        copy_value_dist = value_dist.data.cpu().numpy()
        copy_rewards = rewards.data.cpu().numpy()
        copy_done = (1-done).data.cpu().numpy().astype(np.bool)
        batch_size = self.batch_size
        n_atoms = self.num_atoms
        projected_dist = np.zeros((batch_size, n_atoms), dtype=np.float32)

        # calculate the next state value for each atom in the support set
        for atom in range(n_atoms):
            atom_ = copy_rewards + (self.value_min + atom * self.delta_z) * self.gamma
            tz_j = np.clip(atom_, a_max=self.value_max, a_min=self.value_min)
            # compute where the next value is on the indexes axis of the support set
            b_j = (tz_j - self.value_min) / self.delta_z
            # compute floor and ceiling indexes of the next value on the support set
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            # since l and u are floor and ceiling indexes of the next value on the support set
            # their difference is always 0 at the boundary and 1 otherwise
            # thus, the predicted probability of the next value is distributed proportional to
            #       the difference between the projected value index (b_j) and its floor or ceiling
            # boundary case, floor == ceiling
            eq_mask = (u == l)  # this line gives an array of boolean masks
            projected_dist[eq_mask, l[eq_mask]] += copy_value_dist[eq_mask, atom]
            # otherwise, ceiling - floor == 1, i.e., (u - b_j) + (b_j - l) == 1
            ne_mask = (u != l)
            projected_dist[ne_mask, l[ne_mask]] += copy_value_dist[ne_mask, atom] * (u - b_j)[ne_mask]
            projected_dist[ne_mask, u[ne_mask]] += copy_value_dist[ne_mask, atom] * (b_j - l)[ne_mask]

        # check if a terminal state exists
        if copy_done.any():
            projected_dist[copy_done] = 0.0
            # value at a terminal state should equal to the immediate reward only
            tz_j = np.clip(copy_rewards[copy_done], a_max=self.value_max, a_min=self.value_min)
            b_j = (tz_j - self.value_min) / self.delta_z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = (u == l)
            eq_dones = copy_done.copy()
            eq_dones[copy_done] = eq_mask
            # the value probability is only set to 1.0
            #       when it is a terminal state and its floor and ceiling indexes are the same
            if eq_dones.any():
                projected_dist[eq_dones, l[eq_mask]] = 1.0
            ne_mask = (u != l)
            ne_dones = copy_done.copy()
            ne_dones[copy_done] = ne_mask
            # the value probability is only distributed while summed to 1.0
            #       when it is a terminal state and its floor and ceiling indexes differ by 1 index
            if ne_dones.any():
                projected_dist[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
                projected_dist[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]

        return projected_dist
