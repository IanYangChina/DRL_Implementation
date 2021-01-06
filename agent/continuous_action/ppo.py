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

        # redundant args for compatibility with agent base class (not used by PPO)
        algo_params['prioritised'] = False
        algo_params['discard_time_limit'] = False
        algo_params['tau'] = 1.0
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
            'critic': Critic(self.state_dim, 1).to(self.device),
        })
        self.network_keys_to_save = ['actor', 'critic']
        self.actor_optimizer = Adam(self.network_dict['actor'].parameters(), lr=self.learning_rate)
        self.critic_optimizer = Adam(self.network_dict['critic'].parameters(), lr=self.learning_rate)
        # ppo args
        self.clip_epsilon = algo_params['clip_epsilon']
        self.value_loss_weight = algo_params['value_loss_weight']
        self.GAE_lambda = algo_params['GAE_lambda']
        self.entropy_loss_weight = algo_params['entropy_loss_weight']
        # training args
        # ppo updates network when a batch is available, and discards the data after optimization step
        self.update_interval = self.batch_size
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
            action = self.network_dict['old_actor'].get_action(inputs, mean_pi=test).detach().cpu().numpy()
            return np.clip(action, -self.action_max, self.action_max), None
        else:
            action, probs = self.network_dict['old_actor'].get_action(inputs, probs=True)
            action = action[0].detach().cpu().numpy()
            probs = probs[0].detach().cpu().numpy()
            return np.clip(action, -self.action_max, self.action_max), probs

    def _learn(self, steps=None):
        if len(self.buffer) < self.batch_size:
            return
        if steps is None:
            steps = self.optimizer_steps

        # todo: change the sample method
        batch = self.buffer.full_memory

        actor_inputs = self.normalizer(batch.state)
        actor_inputs = T.tensor(actor_inputs, dtype=T.float32).to(self.device)
        actor_inputs_ = self.normalizer(batch.next_state)
        actor_inputs_ = T.tensor(actor_inputs_, dtype=T.float32).to(self.device)
        actions = T.tensor(batch.action, dtype=T.float32).to(self.device)
        log_probs = T.tensor(batch.log_prob, dtype=T.float32).to(self.device)
        rewards = T.tensor(batch.reward, dtype=T.float32).unsqueeze(1).to(self.device)
        done = T.tensor(batch.done, dtype=T.float32).unsqueeze(1).to(self.device)

        # compute N-step returns & Generalized Advantage Estimator
        #   see https://arxiv.org/pdf/2006.05990.pdf Appendix B.2
        reversed_rewards = rewards.flip(0)
        reversed_next_states = actor_inputs_.flip(0)
        reversed_done = done.flip(0)
        n_step_returns = []
        GAE_returns = []
        discounted_return = 0
        cumulated_step_count = 1
        last_next_state = reversed_next_states[0]
        last_next_state_value = self.network_dict['critic'](last_next_state)
        discounted_GAE_return = 0
        for i in range(reversed_rewards.shape[0]):
            # done flags are stored as 0/1 integers, where 0 represents a done state
            # reset discounted return and cumulated steps when encounter a [True] done flag
            # recalculate the last next state value
            if reversed_done[i] == 0:
                discounted_return = 0
                cumulated_step_count = 1
                last_next_state_value = self.network_dict['critic'](reversed_next_states[i])
                discounted_GAE_return = 0
            discounted_return = reversed_rewards[i] + self.gamma * discounted_return
            # add a discounted next state value to the n-step return
            discount_rate = np.power(self.gamma, cumulated_step_count)
            n_step_return = discounted_return + discount_rate*last_next_state_value.detach()
            cumulated_step_count += 1
            # insert n-step returns top-down
            n_step_returns.insert(0, n_step_return)
            # calculate GAE value
            GAE_discount_rate = np.power(self.GAE_lambda, cumulated_step_count-1)
            discounted_GAE_return += GAE_discount_rate * n_step_return
            GAE_returns.insert(0, (1-self.GAE_lambda)*discounted_GAE_return)

        n_step_returns = T.tensor(n_step_returns, dtype=T.float32).unsqueeze(1).to(self.device)
        GAE_returns = T.tensor(GAE_returns, dtype=T.float32).unsqueeze(1).to(self.device)

        for step in range(steps):
            log_probs_, entropy = self.network_dict['actor'].get_log_probs(actor_inputs, actions)
            state_values = self.network_dict['critic'](actor_inputs)

            ratio = T.exp(log_probs_ - log_probs.detach())
            GAE = GAE_returns - state_values.detach()

            L_clip = T.min(
                ratio*GAE,
                T.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)*GAE
            )
            loss = -(L_clip -
                     self.value_loss_weight*F.mse_loss(state_values, n_step_returns) +
                     self.entropy_loss_weight*entropy.mean())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.mean().backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            self.statistic_dict['critic_loss'].append(F.mse_loss(state_values, n_step_returns).detach().mean().cpu().numpy().item())
            self.statistic_dict['actor_loss'].append(loss.detach().mean().cpu().numpy().item())
            self.statistic_dict['clipped_loss'].append(L_clip.detach().mean().cpu().numpy().item())
            self.statistic_dict['policy_entropy'].append(-log_probs_.detach().mean().cpu().numpy().item())

        self.network_dict['old_actor'].load_state_dict(self.network_dict['actor'].state_dict())
        self.buffer.clear_memory()
