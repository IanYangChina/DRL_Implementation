import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from agent.utils.networks import StochasticActor, Critic
from agent.agent_base import Agent
from agent.utils.exploration_strategy import ConstantChance


class SAC(Agent):
    def __init__(self, algo_params, env, transition_tuple=None, path=None, seed=-1):
        # environment
        self.env = env
        self.env.seed(seed)
        obs = self.env.reset()
        algo_params.update({'state_dim': obs.shape[0],
                            'action_dim': self.env.action_space.shape[0],
                            'action_max': self.env.action_space.high,
                            'init_input_means': None,
                            'init_input_vars': None
                            })
        # training args
        self.training_episodes = algo_params['training_episodes']
        self.testing_gap = algo_params['testing_gap']
        self.testing_episodes = algo_params['testing_episodes']
        self.saving_gap = algo_params['saving_gap']

        super(SAC, self).__init__(algo_params,
                                   transition_tuple=transition_tuple,
                                   goal_conditioned=False,
                                   path=path,
                                   seed=seed)
        # torch
        self.network_dict.update({
            'actor': StochasticActor(self.state_dim, self.action_dim, log_std_min=-6, log_std_max=1).to(self.device),
            'critic_1': Critic(self.state_dim + self.action_dim, 1).to(self.device),
            'critic_1_target': Critic(self.state_dim + self.action_dim, 1).to(self.device),
            'critic_2': Critic(self.state_dim + self.action_dim, 1).to(self.device),
            'critic_2_target': Critic(self.state_dim + self.action_dim, 1).to(self.device),
            'alpha': algo_params['alpha'],
            'log_alpha': T.tensor(np.log(algo_params['alpha']), requires_grad=True, device=self.device),
        })
        self.network_keys_to_save = ['actor', 'critic_1_target']
        self.actor_optimizer = Adam(self.network_dict['actor'].parameters(), lr=self.learning_rate)
        self.critic_1_optimizer = Adam(self.network_dict['critic_1'].parameters(), lr=self.learning_rate)
        self.critic_2_optimizer = Adam(self.network_dict['critic_2'].parameters(), lr=self.learning_rate)
        self._soft_update(self.network_dict['critic_1'], self.network_dict['critic_1_target'], tau=1)
        self._soft_update(self.network_dict['critic_2'], self.network_dict['critic_2_target'], tau=1)
        self.target_entropy = -self.action_dim
        self.alpha_optimizer = Adam([self.network_dict['log_alpha']], lr=self.learning_rate)
        # training args
        self.update_interval = algo_params['update_interval']
        self.actor_update_interval = algo_params['actor_update_interval']
        self.critic_target_update_interval = algo_params['critic_target_update_interval']
        # statistic dict
        self.statistic_dict.update({
            'episode_return': [],
            'episode_test_return': [],
            'alpha': [],
            'policy_entropy': [],
        })

    def run(self, test=False, render=False, load_network_ep=None):
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
            ep_return = self._interact(render, test)
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
            self._save_statistics()
            self._plot_statistics()
        else:
            print("Finished testing")

    def _interact(self, render=False, test=False):
        done = False
        obs = self.env.reset()
        ep_return = 0
        step = 0
        # start a new episode
        while not done:
            if render:
                self.env.render()
            action = self._select_action(obs, test=test)
            new_obs, reward, done, info = self.env.step(action)
            ep_return += reward
            if not test:
                self._remember(obs, action, new_obs, reward, 1 - int(done))
                self.normalizer.store_history(new_obs)
                self.normalizer.update_mean()
                if (step % self.update_interval == 0) and (step != 0):
                    self._learn()
            obs = new_obs
            step += 1
        return ep_return

    def _select_action(self, obs, test=False):
        inputs = self.normalizer(obs)
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        return self.network_dict['actor'].get_action(inputs, mean_pi=test).detach().cpu().numpy()

    def _learn(self, steps=None):
        if len(self.buffer) < self.batch_size:
            return
        if steps is None:
            steps = self.optimizer_steps

        for i in range(steps):
            if self.prioritised:
                batch, weights, inds = self.buffer.sample(self.batch_size)
                weights = T.tensor(weights).view(self.batch_size, 1).to(self.device)
            else:
                batch = self.buffer.sample(self.batch_size)
                weights = T.ones(size=(self.batch_size, 1)).to(self.device)
                inds = None

            actor_inputs = self.normalizer(batch.state)
            actor_inputs = T.tensor(actor_inputs, dtype=T.float32).to(self.device)
            actions = T.tensor(batch.action, dtype=T.float32).to(self.device)
            critic_inputs = T.cat((actor_inputs, actions), dim=1).to(self.device)
            actor_inputs_ = self.normalizer(batch.next_state)
            actor_inputs_ = T.tensor(actor_inputs_, dtype=T.float32).to(self.device)
            rewards = T.tensor(batch.reward, dtype=T.float32).unsqueeze(1).to(self.device)
            done = T.tensor(batch.done, dtype=T.float32).unsqueeze(1).to(self.device)

            if self.discard_time_limit:
                done = done * 0 + 1

            with T.no_grad():
                actions_, log_probs_ = self.network_dict['actor'].get_action(actor_inputs_, probs=True)
                critic_inputs_ = T.cat((actor_inputs_, actions_), dim=1).to(self.device)
                value_1_ = self.network_dict['critic_1_target'](critic_inputs_)
                value_2_ = self.network_dict['critic_2_target'](critic_inputs_)
                value_ = T.min(value_1_, value_2_) - (self.network_dict['alpha'] * log_probs_)
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

            self.statistic_dict['critic_loss'].append(critic_loss.detach().mean().cpu().numpy().item())

            if self.step_count % self.critic_target_update_interval == 0:
                self._soft_update(self.network_dict['critic_1'], self.network_dict['critic_1_target'])
                self._soft_update(self.network_dict['critic_2'], self.network_dict['critic_2_target'])

            if self.step_count % self.actor_update_interval == 0:
                self.actor_optimizer.zero_grad()
                new_actions, new_log_probs = self.network_dict['actor'].get_action(actor_inputs, probs=True)
                critic_eval_inputs = T.cat((actor_inputs, new_actions), dim=1).to(self.device)
                new_values = T.min(self.network_dict['critic_1'](critic_eval_inputs),
                                   self.network_dict['critic_2'](critic_eval_inputs))
                actor_loss = (self.network_dict['alpha']*new_log_probs - new_values).mean()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.alpha_optimizer.zero_grad()
                alpha_loss = (self.network_dict['log_alpha'] * (-new_log_probs - self.target_entropy).detach()).mean()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.network_dict['alpha'] = self.network_dict['log_alpha'].exp()

                self.statistic_dict['actor_loss'].append(actor_loss.detach().mean().cpu().numpy().item())
                self.statistics['alpha'].append(self.network_dict['alpha'].detach().cpu().numpy().item())
                self.statistics['policy_entropy'].append(-new_log_probs.detach().mean().cpu().numpy().item())

            self.step_count += 1