import time
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from ..utils.networks_mlp import Actor, Critic
from ..agent_base import Agent
from ..utils.exploration_strategy import GaussianNoise


class TD3(Agent):
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

        super(TD3, self).__init__(algo_params,
                                   transition_tuple=transition_tuple,
                                   goal_conditioned=False,
                                   path=path,
                                   seed=seed)
        # torch
        self.network_dict.update({
            'actor': Actor(self.state_dim, self.action_dim, action_scaling=self.action_scaling).to(self.device),
            'actor_target': Actor(self.state_dim, self.action_dim, action_scaling=self.action_scaling).to(self.device),
            'critic_1': Critic(self.state_dim + self.action_dim, 1).to(self.device),
            'critic_1_target': Critic(self.state_dim + self.action_dim, 1).to(self.device),
            'critic_2': Critic(self.state_dim + self.action_dim, 1).to(self.device),
            'critic_2_target': Critic(self.state_dim + self.action_dim, 1).to(self.device)
        })
        self.network_keys_to_save = ['actor_target', 'critic_1_target']
        self.actor_optimizer = Adam(self.network_dict['actor'].parameters(), lr=self.actor_learning_rate)
        self._soft_update(self.network_dict['actor'], self.network_dict['actor_target'], tau=1)
        self.critic_1_optimizer = Adam(self.network_dict['critic_1'].parameters(), lr=self.critic_learning_rate)
        self._soft_update(self.network_dict['critic_1'], self.network_dict['critic_1_target'], tau=1)
        self.critic_2_optimizer = Adam(self.network_dict['critic_2'].parameters(), lr=self.critic_learning_rate)
        self._soft_update(self.network_dict['critic_2'], self.network_dict['critic_2_target'], tau=1)
        # behavioural policy args (exploration)
        self.exploration_strategy = GaussianNoise(self.action_dim, self.action_max, mu=0, sigma=0.1)
        # training args
        self.target_noise = algo_params['target_noise']
        self.noise_clip = algo_params['noise_clip']
        self.warmup_step = algo_params['warmup_step']
        self.actor_update_interval = algo_params['actor_update_interval']
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
            obs = new_obs
            self.env_step_count += 1
        return ep_return

    def _select_action(self, obs, test=False):
        obs = self.normalizer(obs)
        with T.no_grad():
            inputs = T.as_tensor(obs, dtype=T.float, device=self.device)
            action = self.network_dict['actor_target'](inputs).detach().cpu().numpy()
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
            rewards = T.as_tensor(batch.reward, dtype=T.float32, device=self.device).unsqueeze(1)
            done = T.as_tensor(batch.done, dtype=T.float32, device=self.device).unsqueeze(1)

            if self.discard_time_limit:
                done = done * 0 + 1

            with T.no_grad():
                actions_ = self.network_dict['actor_target'](actor_inputs_)
                # add noise
                noise = (T.randn_like(actions_, device=self.device) * self.target_noise)
                actions_ += noise.clamp(-self.noise_clip, self.noise_clip)
                actions_ = actions_.clamp(-self.action_max[0], self.action_max[0])
                critic_inputs_ = T.cat((actor_inputs_, actions_), dim=1)
                value_1_ = self.network_dict['critic_1_target'](critic_inputs_)
                value_2_ = self.network_dict['critic_2_target'](critic_inputs_)
                value_ = T.min(value_1_, value_2_)
                value_target = rewards + done * self.gamma * value_

            self.critic_1_optimizer.zero_grad()
            value_estimate_1 = self.network_dict['critic_1'](critic_inputs)
            critic_loss_1 = F.mse_loss(value_estimate_1, value_target.detach(), reduction='none')
            (critic_loss_1 * weights).mean().backward()
            self.critic_1_optimizer.step()

            if self.prioritised:
                assert inds is not None
                self.buffer.update_priority(inds, np.abs(critic_loss_1.cpu().detach().numpy()))

            self.critic_2_optimizer.zero_grad()
            value_estimate_2 = self.network_dict['critic_2'](critic_inputs)
            critic_loss_2 = F.mse_loss(value_estimate_2, value_target.detach(), reduction='none')
            (critic_loss_2 * weights).mean().backward()
            self.critic_2_optimizer.step()

            self.statistic_dict['critic_loss'].append(critic_loss_1.detach().mean())

            if self.optim_step_count % self.actor_update_interval == 0:
                self.actor_optimizer.zero_grad()
                new_actions = self.network_dict['actor'](actor_inputs)
                critic_eval_inputs = T.cat((actor_inputs, new_actions), dim=1)
                actor_loss = -self.network_dict['critic_1'](critic_eval_inputs).mean()
                actor_loss.backward()
                self.actor_optimizer.step()

                self._soft_update(self.network_dict['actor'], self.network_dict['actor_target'])
                self._soft_update(self.network_dict['critic_1'], self.network_dict['critic_1_target'])
                self._soft_update(self.network_dict['critic_2'], self.network_dict['critic_2_target'])

                self.statistic_dict['actor_loss'].append(actor_loss.detach().mean())

            self.optim_step_count += 1
