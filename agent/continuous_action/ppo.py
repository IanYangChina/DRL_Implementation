import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from agent.utils.networks import StochasticActor, Critic
from agent.agent_base import Agent
from collections import namedtuple
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class PPO(Agent):
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

        # redundant args for compatibility with agent base class (not used by PPO)
        algo_params['prioritised'] = False
        algo_params['discard_time_limit'] = False
        algo_params['tau'] = 1.0
        if transition_tuple is None:
            # ppo needs to store historical log probabilities in the buffer
            #   it is convenient to also store state values
            transition_tuple = namedtuple("transition", ('state', 'action', 'log_prob', 'state_value',
                                                         'next_state', 'reward', 'done'))
        super(PPO, self).__init__(algo_params,
                                   transition_tuple=transition_tuple,
                                   goal_conditioned=False,
                                   path=path,
                                   seed=seed)
        # torch
        self.network_dict.update({
            'actor': StochasticActor(self.state_dim, self.action_dim, log_std_min=-6, log_std_max=1, action_scaling=self.action_scaling).to(self.device),
            'old_actor': StochasticActor(self.state_dim, self.action_dim, log_std_min=-6, log_std_max=1, action_scaling=self.action_scaling).to(self.device),
            'critic': Critic(self.state_dim, 1).to(self.device),
        })
        self.network_keys_to_save = ['actor', 'critic']
        self.actor_optimizer = Adam(self.network_dict['actor'].parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = Adam(self.network_dict['critic'].parameters(), lr=self.critic_learning_rate)
        # ppo args
        self.clip_epsilon = algo_params['clip_epsilon']
        self.value_loss_weight = algo_params['value_loss_weight']
        self.GAE_lambda = algo_params['GAE_lambda']
        self.entropy_loss_weight = algo_params['entropy_loss_weight']

        # ppo split a big batch of data into several mini batches to go over for each optimization step (epoch)
        # use a built-in torch sampler to do this
        # see: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/b1152f8e251d827c5c1199aa543fa2739cb12691/a2c_ppo_acktr/storage.py#L107
        # see: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/b1152f8e251d827c5c1199aa543fa2739cb12691/a2c_ppo_acktr/algo/ppo.py#L34
        self.mini_batch_size = algo_params['mini_batch_size']
        assert self.batch_size > self.mini_batch_size
        self.sampler = BatchSampler(
            sampler=SubsetRandomSampler(range(self.batch_size)),
            batch_size=self.mini_batch_size,
            drop_last=True)

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
        # start a new episode
        while not done:
            if render:
                self.env.render()
            action, log_prob, state_value = self._select_action(obs, test=test)
            new_obs, reward, done, info = self.env.step(action)
            ep_return += reward
            if not test:
                done_to_save = done
                if self.env._elapsed_steps == self.env._max_episode_steps:
                    # discard time limit terminal flags, record true terminal flags
                    done_to_save = 1-int(done)
                self._remember(obs, action, log_prob, state_value, new_obs, reward, 1 - int(done_to_save))
                if self.observation_normalization:
                    self.normalizer.store_history(new_obs)
                    self.normalizer.update_mean()
                if (self.env_step_count % self.update_interval == 0) and (self.env_step_count != 0):
                    self._learn()
            obs = new_obs
            self.env_step_count += 1
        return ep_return

    def _select_action(self, obs, test=False):
        inputs = self.normalizer(obs)
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        if test:
            action = self.network_dict['old_actor'].get_action(inputs, mean_pi=test).detach().cpu().numpy()
            return np.clip(action, -self.action_max, self.action_max), None, None
        else:
            action, probs = self.network_dict['old_actor'].get_action(inputs, probs=True)
            action = action[0].detach().cpu().numpy()
            probs = probs[0].detach().cpu().numpy()
            state_value = self.network_dict['critic'](inputs)
            return np.clip(action, -self.action_max, self.action_max), probs, state_value

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
        old_log_probs = T.tensor(batch.log_prob, dtype=T.float32).to(self.device)
        state_values = T.tensor(batch.state_value, dtype=T.float32).to(self.device)
        next_state_values = self.network_dict['critic'](actor_inputs_).detach().view(state_values.size())
        rewards = T.tensor(batch.reward, dtype=T.float32).to(self.device)
        done = T.tensor(batch.done, dtype=T.float32).to(self.device)

        # compute N-step returns & Generalized Advantage Estimator
        #   see https://arxiv.org/pdf/2006.05990.pdf Appendix B.2
        reversed_rewards = T.flip(rewards, dims=[0])
        reversed_state_values = T.flip(state_values, dims=[0])
        reversed_next_state_values = T.flip(next_state_values, dims=[0])
        reversed_done = T.flip(done, dims=[0])
        GAE_returns = []
        gae = 0
        delta = reversed_rewards + reversed_done * self.gamma * reversed_next_state_values - reversed_state_values
        for i in range(reversed_rewards.shape[0]):
            # done flags are stored as 0/1 integers, where 0 represents a terminal next state
            # reset discounted return and cumulated steps when encounter a [True] done flag
            # recalculate the last next state value

            gae = delta[i] + self.gamma * self.GAE_lambda * gae * reversed_done[i]
            gae = gae * reversed_done[i]
            gae_return = gae + reversed_state_values[i]
            GAE_returns.insert(0, gae_return)

        GAE_returns = T.tensor(GAE_returns, dtype=T.float32).unsqueeze(1).to(self.device)
        GAE = GAE_returns - state_values
        # normalize advantages
        GAE = (GAE - GAE.mean()) / (GAE.std() + 1e-5)

        # go over the batch several times
        for step in range(steps):
            # each time split the batch into several mini batches
            for indices in self.sampler:
                mb_actor_inputs = actor_inputs[indices]
                mb_actions = actions[indices]
                mb_old_log_probs = old_log_probs[indices]
                mb_GAE_returns = GAE_returns[indices]

                mb_log_probs_, mb_entropy = self.network_dict['actor'].get_log_probs(mb_actor_inputs, mb_actions)
                mb_state_values = self.network_dict['critic'](mb_actor_inputs)

                mb_ratio = T.exp(mb_log_probs_ - mb_old_log_probs.detach())
                mb_GAE = GAE[indices]

                L_clip = T.min(
                    mb_ratio*mb_GAE,
                    T.clamp(mb_ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)*mb_GAE
                )
                value_loss = F.mse_loss(mb_state_values, mb_GAE_returns)
                loss = -(L_clip -
                         self.value_loss_weight*value_loss +
                         self.entropy_loss_weight*mb_entropy.mean())

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.mean().backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                self.statistic_dict['critic_loss'].append(value_loss.detach().mean().cpu().numpy().item())
                self.statistic_dict['actor_loss'].append(loss.detach().mean().cpu().numpy().item())
                self.statistic_dict['clipped_loss'].append(L_clip.detach().mean().cpu().numpy().item())
                self.statistic_dict['policy_entropy'].append(-mb_log_probs_.detach().mean().cpu().numpy().item())

        self.network_dict['old_actor'].load_state_dict(self.network_dict['actor'].state_dict())
        self.buffer.clear_memory()
