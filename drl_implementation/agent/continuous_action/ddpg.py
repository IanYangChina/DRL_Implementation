import time
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from ..utils.networks_mlp import Actor, Critic
from ..agent_base import Agent
from ..utils.exploration_strategy import OUNoise, GaussianNoise


class DDPG(Agent):
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

        super(DDPG, self).__init__(algo_params,
                                   transition_tuple=transition_tuple,
                                   goal_conditioned=False,
                                   path=path,
                                   seed=seed)
        # torch
        self.network_dict.update({
            'actor': Actor(self.state_dim, self.action_dim).to(self.device),
            'actor_target': Actor(self.state_dim, self.action_dim).to(self.device),
            'critic': Critic(self.state_dim + self.action_dim, 1).to(self.device),
            'critic_target': Critic(self.state_dim + self.action_dim, 1).to(self.device)
        })
        self.network_keys_to_save = ['actor_target', 'critic_target']
        self.actor_optimizer = Adam(self.network_dict['actor'].parameters(), lr=self.actor_learning_rate)
        self._soft_update(self.network_dict['actor'], self.network_dict['actor_target'], tau=1)
        self.critic_optimizer = Adam(self.network_dict['critic'].parameters(), lr=self.critic_learning_rate, weight_decay=algo_params['Q_weight_decay'])
        self._soft_update(self.network_dict['critic'], self.network_dict['critic_target'], tau=1)
        # behavioural policy args (exploration)
        self.exploration_strategy = GaussianNoise(self.action_dim, self.action_max, sigma=0.1)
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
                self.statistic_dict['episode_test_return'].append(sum(ep_test_return)/self.testing_episodes)
                print("Episode %i" % ep, "test return %0.1f" % (sum(ep_test_return)/self.testing_episodes))

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
        # self.exploration_strategy.reset()
        # start a new episode
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

            actor_inputs = np.array(self.normalizer(batch.state))
            actor_inputs = T.as_tensor(actor_inputs, dtype=T.float32, device=self.device)
            actions = T.as_tensor(np.array(batch.action), dtype=T.float32, device=self.device)
            critic_inputs = T.cat((actor_inputs, actions), dim=1)
            actor_inputs_ = np.array(self.normalizer(batch.next_state))
            actor_inputs_ = T.as_tensor(np.array(actor_inputs_), dtype=T.float32, device=self.device)
            rewards = T.as_tensor(np.array(batch.reward), dtype=T.float32, device=self.device).unsqueeze(1)
            done = T.as_tensor(np.array(batch.done), dtype=T.float32, device=self.device).unsqueeze(1)

            if self.discard_time_limit:
                done = done * 0 + 1

            with T.no_grad():
                actions_ = self.network_dict['actor_target'](actor_inputs_)
                critic_inputs_ = T.cat((actor_inputs_, actions_), dim=1)
                value_ = self.network_dict['critic_target'](critic_inputs_)
                value_target = rewards + done * self.gamma * value_

            self.critic_optimizer.zero_grad()
            value_estimate = self.network_dict['critic'](critic_inputs)
            critic_loss = F.mse_loss(value_estimate, value_target, reduction='none')
            (critic_loss * weights).mean().backward()
            self.critic_optimizer.step()

            if self.prioritised:
                assert inds is not None
                self.buffer.update_priority(inds, np.abs(critic_loss.cpu().detach().numpy()))

            self.actor_optimizer.zero_grad()
            new_actions = self.network_dict['actor'](actor_inputs)
            critic_eval_inputs = T.cat((actor_inputs, new_actions), dim=1).to(self.device)
            actor_loss = -self.network_dict['critic'](critic_eval_inputs).mean()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.network_dict['actor'], self.network_dict['actor_target'])
            self._soft_update(self.network_dict['critic'], self.network_dict['critic_target'])

            self.statistic_dict['critic_loss'].append(critic_loss.detach().mean())
            self.statistic_dict['actor_loss'].append(actor_loss.detach().mean())
