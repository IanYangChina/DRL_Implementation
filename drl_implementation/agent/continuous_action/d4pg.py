import time
import queue
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from ..utils.networks_mlp import Actor, Critic
from ..distributed_agent_base import CentralProcessor, Worker, Learner
from ..utils.exploration_strategy import GaussianNoise


class D4PG(CentralProcessor):
    def __init__(self, algo_params, env_name, env_source, path=None, worker_seeds=None, seed=0):
        super(D4PG, self).__init__(algo_params=algo_params,
                                   env_name=env_name,
                                   env_source=env_source,
                                   learner=D4PGLearner,
                                   worker=D4PGWorker,
                                   path=path,
                                   worker_seeds=worker_seeds,
                                   seed=seed)


class D4PGWorker(Worker):
    def __init__(self, algo_params, env, queues, path=None, seed=0, i=0):
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
        self.saving_gap = algo_params['saving_gap']
        super(D4PGWorker, self).__init__(algo_params, queues, path=path, seed=seed, i=i)

        # networks
        self.num_atoms = algo_params['num_atoms']
        self.value_max = algo_params['value_max']
        self.value_min = algo_params['value_min']
        self.delta_z = (self.value_max - self.value_min) / (self.num_atoms - 1)
        self.reward_scaling = algo_params['reward_scaling']
        self.network_dict.update({
            'actor_target': Actor(self.state_dim, self.action_dim).to(self.device),
            'critic_target': Critic(self.state_dim + self.action_dim, self.num_atoms, softmax=True).to(self.device)
        })
        self.network_keys_to_save = ['actor_target', 'critic_target']

        # whether to store transition with pre-computed td-error or not
        self.store_with_given_priority = algo_params['store_with_given_priority']
        # make sure the worker networks are synced with the initial learner
        synced = False
        while not synced:
            synced = self._download_actor_networks(keys=['actor_target', 'critic_target'])

        # behavioural policy (exploration)
        self.exploration_strategy = GaussianNoise(self.action_dim,
                                                  self.action_max,
                                                  scale=algo_params['gaussian_scale'],
                                                  sigma=algo_params['gaussian_sigma'])

        # statistic dict
        self.statistic_dict.update({
            'episode_return': [],
            'step_test_return': []
        })

    def run(self, render=False, test=False, load_network_ep=None, sleep=0):
        print('Worker No. {} on'.format(self.worker_id))
        num_episode = self.training_episodes
        for ep in range(num_episode):
            self.queues['global_episode_count'].value += 1
            ep_return = self._interact(render, test, sleep=sleep)
            self.statistic_dict['episode_return'].append(ep_return)
            print("'Worker No. %i" % self.worker_id, "Episode %i" % ep, "return %0.1f" % ep_return)
            if (ep % self.saving_gap == 0) and (ep != 0):
                self._save_network(ep=ep)
            if (ep % self.worker_update_gap == 0) and (ep != 0):
                synced = False
                while not synced:
                    synced = self._download_actor_networks(keys=self.network_keys_to_save)
        print("Saving Worker No. {} statistics...".format(self.worker_id))
        self._save_statistics()
        self._plot_statistics(keys=['episode_return'])
        print('Worker No. {} shutdown'.format(self.worker_id))

    def _interact(self, render=False, test=False, sleep=0):
        done = False
        obs = self.env.reset()
        ep_return = 0
        while not done:
            if render:
                self.env.render()
            action = self._select_action(obs, test=test)
            new_obs, reward, done, info = self.env.step(action)
            reward = reward * self.reward_scaling
            time.sleep(sleep)
            ep_return += reward
            if not test:
                if self.store_with_given_priority:
                    # compute td-error using local network for better initial priority
                    # see: https://arxiv.org/pdf/1803.00933.pdf
                    with T.no_grad():
                        critic_input = T.from_numpy(np.concatenate((obs, action), axis=0)).type(T.float).unsqueeze(0).to(self.device)
                        new_action = self.network_dict['actor_target'](T.tensor(new_obs, dtype=T.float).to(self.device)).cpu().numpy()
                        critic_input_ = T.from_numpy(np.concatenate((new_obs, new_action), axis=0)).type(T.float).unsqueeze(0).to(self.device)
                        value_dist = self.network_dict['critic_target'](critic_input)
                        value_dist_ = self.network_dict['critic_target'](critic_input_)
                        value_dist_target = project_value_distribution(value_dist_, reward, 1 - int(done), self.num_atoms, self.value_max, self.value_min, self.delta_z, self.gamma)
                        value_dist_target = T.from_numpy(value_dist_target).type(T.float).to(self.device)
                        td_error = F.binary_cross_entropy(value_dist, value_dist_target, reduction='none').sum(dim=1).cpu().numpy()
                        priority = np.abs(td_error)
                    self._remember({
                            'priority': priority,
                            'transition': (obs, action, new_obs, reward, 1 - int(done))
                        })
                else:
                    self._remember((obs, action, new_obs, reward, 1 - int(done)))
                if self.observation_normalization:
                    # currently observation normalization is off
                    self.normalizer.store_history(new_obs)
                    self.normalizer.update_mean()
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


class D4PGLearner(Learner):
    def __init__(self, algo_params, env, queues, path=None, seed=0):
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
        self.env.close()

        super(D4PGLearner, self).__init__(algo_params, queues, path=path, seed=seed)
        # categorical distribution atoms
        self.num_atoms = algo_params['num_atoms']
        self.value_max = algo_params['value_max']
        self.value_min = algo_params['value_min']
        self.delta_z = (self.value_max - self.value_min) / (self.num_atoms - 1)
        self.support = T.linspace(self.value_min, self.value_max, steps=self.num_atoms, device=self.device)

        self.network_dict.update({
            'actor': Actor(self.state_dim, self.action_dim).to(self.device),
            'actor_target': Actor(self.state_dim, self.action_dim).to(self.device),
            'critic': Critic(self.state_dim + self.action_dim, self.num_atoms, softmax=True).to(self.device),
            'critic_target': Critic(self.state_dim + self.action_dim, self.num_atoms, softmax=True).to(self.device)
        })
        self.actor_optimizer = Adam(self.network_dict['actor'].parameters(), lr=self.actor_learning_rate)
        self._soft_update(self.network_dict['actor'], self.network_dict['actor_target'], tau=1)
        self.critic_optimizer = Adam(self.network_dict['critic'].parameters(), lr=self.critic_learning_rate,
                                     weight_decay=algo_params['Q_weight_decay'])
        self._soft_update(self.network_dict['critic'], self.network_dict['critic_target'], tau=1)
        self._upload_learner_networks(keys=['actor_target', 'critic_target'])

    def run(self):
        print('Learner on')
        while self.queues['learner_step_count'].value < self.learner_steps:
            try:
                batch = self.queues['batch_queue'].get_nowait()
            except queue.Empty:
                # print("empty batch queue")
                continue
            self._learn(batch=batch)
            if self.queues['learner_step_count'].value % self.learner_upload_gap == 0:
                self._upload_learner_networks(keys=['actor_target', 'critic_target'])
        print("Saving learner statistics...")
        self._plot_statistics(keys=['actor_loss', 'critic_loss'], save_to_file=True)
        print('Learner shutdown')

    def _learn(self, steps=None, batch=None):
        if batch is None:
            return
        if steps is None:
            steps = self.optimizer_steps

        for i in range(steps):
            if self.prioritised:
                state, action, next_state, reward, done, weights, inds = batch
                weights = T.as_tensor(weights, device=self.device).view(self.batch_size, 1)
            else:
                state, action, next_state, reward, done = batch
                weights = T.ones(size=(self.batch_size, 1), device=self.device)
                inds = None

            actor_inputs = self.normalizer(state)
            actor_inputs = T.as_tensor(actor_inputs, dtype=T.float32, device=self.device)
            actions = T.as_tensor(action, dtype=T.float32, device=self.device)
            critic_inputs = T.cat((actor_inputs, actions), dim=1)
            actor_inputs_ = self.normalizer(next_state)
            actor_inputs_ = T.as_tensor(actor_inputs_, dtype=T.float32, device=self.device)
            rewards = T.as_tensor(reward, dtype=T.float32, device=self.device)
            done = T.as_tensor(done, dtype=T.float32, device=self.device)

            if self.discard_time_limit:
                done = done * 0 + 1

            with T.no_grad():
                actions_ = self.network_dict['actor_target'](actor_inputs_)
                critic_inputs_ = T.cat((actor_inputs_, actions_), dim=1)
                value_dist_ = self.network_dict['critic_target'](critic_inputs_)
                value_dist_target = project_value_distribution(value_dist_, rewards, done, self.num_atoms, self.value_max, self.value_min, self.delta_z, self.gamma)
                value_dist_target = T.as_tensor(value_dist_target, device=self.device)

            self.critic_optimizer.zero_grad()
            value_dist_estimate = self.network_dict['critic'](critic_inputs)
            critic_loss = F.binary_cross_entropy(value_dist_estimate, value_dist_target, reduction='none').sum(dim=1)
            (critic_loss * weights).mean().backward()
            self.critic_optimizer.step()

            if self.prioritised:
                try:
                    self.queues['priority_queue'].put((inds, np.abs(critic_loss.cpu().detach().numpy())))
                except queue.Full:
                    pass

            self.actor_optimizer.zero_grad()
            new_actions = self.network_dict['actor'](actor_inputs)
            critic_eval_inputs = T.cat((actor_inputs, new_actions), dim=1).to(self.device)
            # take the expectation of the value distribution as the policy loss
            actor_loss = -(self.network_dict['critic'](critic_eval_inputs) * self.support)
            actor_loss = actor_loss.sum(dim=1)
            actor_loss.mean().backward()
            self.actor_optimizer.step()

            self._soft_update(self.network_dict['actor'], self.network_dict['actor_target'])
            self._soft_update(self.network_dict['critic'], self.network_dict['critic_target'])

            self.statistic_dict['critic_loss'].append(critic_loss.detach().mean())
            self.statistic_dict['actor_loss'].append(actor_loss.detach().mean())

            self.queues['learner_step_count'].value += 1


def project_value_distribution(value_dist, rewards, done, n_atoms, value_max, value_min, delta_z, gamma):
    # refer to https://github.com/schatty/d4pg-pytorch/blob/7dc23096a45bc4036fbb02493e0b052d57cfe4c6/models/d4pg/l2_projection.py#L7
    # comments added
    copy_value_dist = value_dist.data.cpu().numpy()
    if isinstance(rewards, T.Tensor):
        copy_rewards = rewards.data.cpu().numpy()
        copy_done = (1 - done).data.cpu().numpy().astype(np.bool)
    else:
        copy_rewards = np.array(rewards).reshape(-1)
        copy_done = np.array((1 - done)).reshape(-1).astype(np.bool)
    batch_size = copy_value_dist.shape[0]
    projected_dist = np.zeros((batch_size, n_atoms), dtype=np.float32)

    for atom in range(n_atoms):
        tz_j = np.clip(copy_rewards + (value_min + atom * delta_z) * gamma, a_max=value_max, a_min=value_min)
        # compute where the next value is on the indexes axis of the support set
        b_j = (tz_j - value_min) / delta_z
        # compute floor and ceiling indexes of the next value on the support set
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        # since l and u are floor and ceiling indexes of the next value on the support set, their difference is always 0 at the boundary and 1 otherwise
        # thus, the predicted probability of the next value is distributed proportional to the difference between the projected value index (b_j) and its floor or ceiling
        # boundary case, floor == ceiling
        eq_mask = u == l
        projected_dist[eq_mask, l[eq_mask]] += copy_value_dist[eq_mask, atom]
        ne_mask = u != l
        # otherwise, (u - b_j) + (b_j - l) == 1
        projected_dist[ne_mask, l[ne_mask]] += copy_value_dist[ne_mask, atom] * (u - b_j)[ne_mask]
        projected_dist[ne_mask, u[ne_mask]] += copy_value_dist[ne_mask, atom] * (b_j - l)[ne_mask]

    # check if a terminal state exists
    if copy_done.any():
        projected_dist[copy_done] = 1.0
        # value at a terminal state should equal to the immediate reward only
        tz_j = np.clip(copy_rewards[copy_done], a_max=value_max, a_min=value_min)
        b_j = (tz_j - value_min) / delta_z
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
