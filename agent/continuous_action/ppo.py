import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from agent.utils.networks import StochasticActor, Critic
from agent.agent_base import Agent
from collections import namedtuple


class PPO(Agent):
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

        # ppo does not use prioritised replay
        algo_params['prioritised'] = False
        if transition_tuple is None:
            # ppo needs to store historical log probabilities in the buffer
            transition_tuple = namedtuple("transition", ('state', 'action', 'log_prob', 'next_state', 'reward', 'done'))
        super(PPO, self).__init__(algo_params,
                                   transition_tuple=transition_tuple,
                                   goal_conditioned=False,
                                   path=path,
                                   seed=seed)
        # torch
        self.network_dict.update({
            'actor': StochasticActor(self.state_dim, self.action_dim, log_std_min=-6, log_std_max=1).to(self.device),
            'old_actor': StochasticActor(self.state_dim, self.action_dim, log_std_min=-6, log_std_max=1).to(self.device),
            'critic': Critic(self.state_dim + self.action_dim, 1).to(self.device),
        })
        self.network_keys_to_save = ['actor', 'critic']
        self.actor_optimizer = Adam(self.network_dict['actor'].parameters(), lr=self.learning_rate)
        self.critic_optimizer = Adam(self.network_dict['critic'].parameters(), lr=self.learning_rate)
        # ppo args
        self.clip_epsilon = algo_params['clip_epsilon']
        self.value_loss_weight = algo_params['value_loss_weight']
        self.return_normalization = algo_params['return_normalization']
        self.GAE_lambda = algo_params['GAE_lambda']
        # training args
        # ppo updates network when a batch is available, and discards the data after optimization step
        self.update_interval = self.batch_size
        self.actor_update_interval = algo_params['actor_update_interval']
        self.critic_target_update_interval = algo_params['critic_target_update_interval']
        # statistic dict
        self.statistic_dict.update({
            'episode_return': [],
            'episode_test_return': [],
            'policy_entropy': [],
            'clipped_loss': []
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
            action, log_prob = self._select_action(obs, test=test)
            new_obs, reward, done, info = self.env.step(action)
            ep_return += reward
            if not test:
                self._remember(obs, action, log_prob, new_obs, reward, 1 - int(done))
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
        if test:
            return self.network_dict['old_actor'].get_action(inputs, mean_pi=test).detach().cpu().numpy()
        else:
            action, probs = self.network_dict['old_actor'].get_action(inputs, probs=True)
            return action[0].detach().cpu().numpy(), probs[0].detach().cpu().numpy()

    def _learn(self, steps=None):
        if len(self.buffer) < self.batch_size:
            return
        if steps is None:
            steps = self.optimizer_steps

        batch = self.buffer.full_memory

        actor_inputs = self.normalizer(batch.state)
        actor_inputs = T.tensor(actor_inputs, dtype=T.float32).to(self.device)
        actor_inputs_ = self.normalizer(batch.next_state)
        actor_inputs_ = T.tensor(actor_inputs_, dtype=T.float32).to(self.device)
        actions = T.tensor(batch.action, dtype=T.float32).to(self.device)
        log_probs = T.tensor(batch.log_prob, dtype=T.float32).to(self.device)
        rewards = T.tensor(batch.reward, dtype=T.float32).unsqueeze(1).to(self.device)
        done = T.tensor(batch.done, dtype=T.float32).unsqueeze(1).to(self.device)

        # compute N-step returns
        returns = []
        discounted_return = 0
        rewards = rewards.flip(0)
        done = done.flip(0)
        for i in range(rewards.shape[0]):
            # done flags are stored as 0/1 integers, where 0 represents a done state
            if done[i] == 0:
                discounted_return = 0
            discounted_return = rewards[i] + self.gamma * discounted_return
            # insert n-step returns top-down
            returns.insert(0, discounted_return)
        returns = T.tensor(returns, dtype=T.float32).unsqueeze(1).to(self.device)
        if self.return_normalization:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        for step in range(steps):
            log_probs_, entropy = self.network_dict['actor'].get_log_probs(actor_inputs, actions)
            state_values = self.network_dict['critic'](actor_inputs)
            next_state_values = self.network_dict['critic'](actor_inputs_)

            ratio = T.exp(log_probs_ - log_probs.detach())
            # no done flag trick
            returns = returns + self.gamma * next_state_values.detach()
            advantages = returns - state_values.detach()

            # compute general advantage esimator
            GAE = []
            gae_t = 0
            advantages.flip(0)
            for i in range(rewards.shape[0]):
                if done[i] == 0:
                    gae_t = 0
                gae_t = advantages[i] + self.GAE_lambda * gae_t
                GAE.insert(0, (1-self.GAE_lambda)*gae_t)
            GAE = T.tensor(GAE, dtype=T.float32).unsqueeze(1).to(self.device)

            L_clip = T.min(
                ratio*GAE,
                T.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)*GAE
            )
            loss = -(L_clip -
                     self.value_loss_weight*F.mse_loss(state_values, returns))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.mean().backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            self.statistic_dict['critic_loss'].append(F.mse_loss(state_values, returns).detach().mean().cpu().numpy().item())
            self.statistic_dict['actor_loss'].append(loss.detach().mean().cpu().numpy().item())
            self.statistic_dict['clipped_loss'].append(L_clip.detach().mean().cpu().numpy().item())
            self.statistic_dict['policy_entropy'].append(-log_probs_.detach().mean().cpu().numpy().item())

        self.network_dict['old_actor'].load_state_dict(self.network_dict['actor'].state_dict())
        self.buffer.clear_memory()
