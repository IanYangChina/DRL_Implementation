import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from agent.utils.networks import Actor, Critic
from agent.agent_base import Agent
from agent.utils.exploration_strategy import ConstantChance


class DDPG(Agent):
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
        self.actor_optimizer = Adam(self.network_dict['actor'].parameters(), lr=self.learning_rate)
        self._soft_update(self.network_dict['actor'], self.network_dict['actor_target'], tau=1)
        self.critic_optimizer = Adam(self.network_dict['critic'].parameters(), lr=self.learning_rate)
        self._soft_update(self.network_dict['critic'], self.network_dict['critic_target'], tau=1)
        # behavioural policy args (exploration)
        self.exploration_strategy = ConstantChance(chance=algo_params['random_action_chance'], rng=self.rng)
        self.noise_deviation = algo_params['noise_deviation']
        # training args
        self.update_interval = algo_params['update_interval']
        # statistic dict
        self.statistic_dict.update({
            'ep_return': []
        })

    def _interact(self, ep, render=False, test=False, load_network_ep=None):
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

        self.statistic_dict['ep_return'].append(ep_return)
        print("Episode %i" % ep, "return %0.1f" % ep_return)

    def _select_action(self, obs, test=False):
        inputs = self.normalizer(obs)
        # evaluate
        if test:
            with T.no_grad():
                inputs = T.tensor(inputs, dtype=T.float).to(self.device)
                action = self.network_dict['actor_target'](inputs).cpu().detach().numpy()
            return np.clip(action, -self.action_max, self.action_max)
        # train
        explore = self.exploration_strategy()
        if explore:
            return self.rng.uniform(-self.action_max, self.action_max, size=(self.action_dim,))
        else:
            with T.no_grad():
                inputs = T.tensor(inputs, dtype=T.float).to(self.device)
                noise = self.noise_deviation * self.rng.standard_normal(size=self.action_dim)
                action = self.network_dict['actor_target'](inputs).cpu().detach().numpy()
                return np.clip(action + noise, -self.action_max, self.action_max)

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
                actions_ = self.network_dict['actor_target'](actor_inputs_)
                critic_inputs_ = T.cat((actor_inputs_, actions_), dim=1).to(self.device)
                value_ = self.network_dict['critic_target'](critic_inputs_)
                value_target = rewards + done * self.gamma * value_

            value_estimate = self.network_dict['critic'](critic_inputs)
            self.critic_optimizer.zero_grad()
            critic_loss = F.mse_loss(value_estimate, value_target, reduction='none')
            (critic_loss * weights).mean().backward()
            self.critic_optimizer.step()
            if self.prioritised:
                assert inds is not None
                self.buffer.update_priority(inds, np.abs(critic_loss.cpu().detach().numpy()))

            new_actions = self.network_dict['actor'](actor_inputs)
            critic_eval_inputs = T.cat((actor_inputs, new_actions), dim=1).to(self.device)
            self.actor_optimizer.zero_grad()
            actor_loss = -self.network_dict['critic'](critic_eval_inputs).mean()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.network_dict['actor'], self.network_dict['actor_target'])
            self._soft_update(self.network_dict['critic'], self.network_dict['critic_target'])

            self.statistic_dict['critic_loss'].append(critic_loss.detach().mean().cpu().numpy().item())
            self.statistic_dict['actor_loss'].append(actor_loss.detach().mean().cpu().numpy().item())
