import time
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from ..utils.networks_mlp import Actor, Critic
from ..agent_base import Agent
from ..utils.exploration_strategy import EGreedyGaussian


class GoalConditionedDDPG(Agent):
    def __init__(self, algo_params, env, transition_tuple=None, path=None, seed=-1):
        # environment
        self.env = env
        self.env.seed(seed)
        obs = self.env.reset()
        algo_params.update({'state_dim': obs['observation'].shape[0],
                            'goal_dim': obs['desired_goal'].shape[0],
                            'action_dim': self.env.action_space.shape[0],
                            'action_max': self.env.action_space.high,
                            'action_scaling': self.env.action_space.high[0],
                            'init_input_means': None,
                            'init_input_vars': None
                            })
        self.curriculum = False
        if 'curriculum' in algo_params.keys():
            self.curriculum = algo_params['curriculum']
        # training args
        self.training_epochs = algo_params['training_epochs']
        self.training_cycles = algo_params['training_cycles']
        self.training_episodes = algo_params['training_episodes']
        self.testing_gap = algo_params['testing_gap']
        self.testing_episodes = algo_params['testing_episodes']
        self.saving_gap = algo_params['saving_gap']

        super(GoalConditionedDDPG, self).__init__(algo_params,
                                                  transition_tuple=transition_tuple,
                                                  goal_conditioned=True,
                                                  path=path,
                                                  seed=seed)
        # torch
        self.network_dict.update({
            'actor': Actor(self.state_dim + self.goal_dim, self.action_dim, action_scaling=self.action_scaling).to(
                self.device),
            'actor_target': Actor(self.state_dim + self.goal_dim, self.action_dim,
                                  action_scaling=self.action_scaling).to(self.device),
            'critic_1': Critic(self.state_dim + self.goal_dim + self.action_dim, 1).to(self.device),
            'critic_1_target': Critic(self.state_dim + self.goal_dim + self.action_dim, 1).to(self.device),
            'critic_2': Critic(self.state_dim + self.goal_dim + self.action_dim, 1).to(self.device),
            'critic_2_target': Critic(self.state_dim + self.goal_dim + self.action_dim, 1).to(self.device),
        })
        self.network_keys_to_save = ['actor_target', 'critic_1_target', 'critic_2_target']
        self.actor_optimizer = Adam(self.network_dict['actor'].parameters(), lr=self.actor_learning_rate)
        self._soft_update(self.network_dict['actor'], self.network_dict['actor_target'], tau=1)
        self.critic_1_optimizer = Adam(self.network_dict['critic_1'].parameters(), lr=self.critic_learning_rate,
                                       weight_decay=algo_params['Q_weight_decay'])
        self._soft_update(self.network_dict['critic_1'], self.network_dict['critic_1_target'], tau=1)
        self.critic_2_optimizer = Adam(self.network_dict['critic_2'].parameters(), lr=self.critic_learning_rate,
                                       weight_decay=algo_params['Q_weight_decay'])
        self._soft_update(self.network_dict['critic_2'], self.network_dict['critic_2_target'], tau=1)
        # behavioural policy args (exploration)
        # different from the original DDPG paper, the HER paper uses another exploration strategy
        #   paper: https://papers.nips.cc/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html
        self.exploration_strategy = EGreedyGaussian(action_dim=self.action_dim,
                                                    action_max=self.action_max,
                                                    chance=algo_params['random_action_chance'],
                                                    sigma=algo_params['noise_deviation'], rng=self.rng)
        self.noise_deviation = algo_params['noise_deviation']
        # training args
        self.clip_value = algo_params['clip_value']
        # statistic dict
        self.statistic_dict.update({
            'cycle_return': [],
            'cycle_success_rate': [],
            'epoch_test_return': [],
            'epoch_test_success_rate': []
        })

    def run(self, test=False, render=False, load_network_ep=None, sleep=0):
        # training setup uses a hierarchy of Epoch, Cycle and Episode
        #   following the HER paper: https://papers.nips.cc/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html
        if test:
            if load_network_ep is not None:
                print("Loading network parameters...")
                self._load_network(ep=load_network_ep)
            print("Start testing...")
        else:
            print("Start training...")

        for epo in range(self.training_epochs):
            if self.curriculum:
                self.env.activate_curriculum_update()
            for cyc in range(self.training_cycles):
                cycle_return = 0
                cycle_success = 0
                for ep in range(self.training_episodes):
                    ep_return = self._interact(render, test, sleep=sleep)
                    cycle_return += ep_return
                    if ep_return > -self.env._max_episode_steps:
                        cycle_success += 1

                self.statistic_dict['cycle_return'].append(cycle_return / self.training_episodes)
                self.statistic_dict['cycle_success_rate'].append(cycle_success / self.training_episodes)
                print("Epoch %i" % epo, "Cycle %i" % cyc,
                      "avg. return %0.1f" % (cycle_return / self.training_episodes),
                      "success rate %0.1f" % (cycle_success / self.training_episodes))

            if (epo % self.testing_gap == 0) and (epo != 0) and (not test):
                if self.curriculum:
                    self.env.deactivate_curriculum_update()
                # testing during training
                test_return = 0
                test_success = 0
                for test_ep in range(self.testing_episodes):
                    ep_test_return = self._interact(render, test=True)
                    test_return += ep_test_return
                    if ep_test_return > -self.env._max_episode_steps:
                        test_success += 1
                self.statistic_dict['epoch_test_return'].append(test_return / self.testing_episodes)
                self.statistic_dict['epoch_test_success_rate'].append(test_success / self.testing_episodes)
                print("Epoch %i" % epo, "test avg. return %0.1f" % (test_return / self.testing_episodes))

            if (epo % self.saving_gap == 0) and (epo != 0) and (not test):
                self._save_network(ep=epo)

        if not test:
            print("Finished training")
            print("Saving statistics...")
            self._plot_statistics(
                x_labels={
                    'critic_loss': 'Optimization epoch (per ' + str(self.optimizer_steps) + ' steps)',
                    'actor_loss': 'Optimization epoch (per ' + str(self.optimizer_steps) + ' steps)'
                },
                save_to_file=True)
        else:
            print("Finished testing")

    def _interact(self, render=False, test=False, sleep=0):
        done = False
        obs = self.env.reset()
        if self.curriculum:
            self.env._max_episode_steps = self.env.env.curriculum_goal_step
        ep_return = 0
        new_episode = True
        # start a new episode
        while not done:
            if render:
                self.env.render()
            action = self._select_action(obs, test=test)
            new_obs, reward, done, info = self.env.step(action)
            time.sleep(sleep)
            ep_return += reward
            if not test:
                self._remember(obs['observation'], obs['desired_goal'], action,
                               new_obs['observation'], new_obs['achieved_goal'], reward, 1 - int(done),
                               new_episode=new_episode)
                if self.observation_normalization:
                    self.normalizer.store_history(np.concatenate((new_obs['observation'],
                                                                  new_obs['achieved_goal']), axis=0))
            obs = new_obs
            new_episode = False
        if not test:
            self.normalizer.update_mean()
            self._learn()
        return ep_return

    def _select_action(self, obs, test=False):
        inputs = np.concatenate((obs['observation'], obs['desired_goal']), axis=0)
        inputs = self.normalizer(inputs)
        with T.no_grad():
            inputs = T.as_tensor(inputs, dtype=T.float, device=self.device)
            action = self.network_dict['actor_target'](inputs).cpu().detach().numpy()
        if test:
            # evaluate
            return np.clip(action, -self.action_max, self.action_max)
        else:
            # explore
            return self.exploration_strategy(action)

    def _learn(self, steps=None):
        if self.hindsight:
            self.buffer.modify_episodes()
        self.buffer.store_episodes()
        if len(self.buffer) < self.batch_size:
            return
        if steps is None:
            steps = self.optimizer_steps

        critic_losses = T.zeros(1, device=self.device)
        actor_losses = T.zeros(1, device=self.device)
        for i in range(steps):
            if self.prioritised:
                batch, weights, inds = self.buffer.sample(self.batch_size)
                weights = T.as_tensor(weights).view(self.batch_size, 1).to(self.device)
            else:
                batch = self.buffer.sample(self.batch_size)
                weights = T.ones(size=(self.batch_size, 1)).to(self.device)
                inds = None

            actor_inputs = np.concatenate((batch.state, batch.desired_goal), axis=1)
            actor_inputs = self.normalizer(actor_inputs)
            actor_inputs = T.as_tensor(actor_inputs, dtype=T.float32, device=self.device)
            actions = T.as_tensor(batch.action, dtype=T.float32, device=self.device)
            critic_inputs = T.cat((actor_inputs, actions), dim=1)
            actor_inputs_ = np.concatenate((batch.next_state, batch.desired_goal), axis=1)
            actor_inputs_ = self.normalizer(actor_inputs_)
            actor_inputs_ = T.as_tensor(actor_inputs_, dtype=T.float32, device=self.device)
            rewards = T.as_tensor(batch.reward, dtype=T.float32, device=self.device).unsqueeze(1)
            done = T.as_tensor(batch.done, dtype=T.float32, device=self.device).unsqueeze(1)

            if self.discard_time_limit:
                done = done * 0 + 1

            with T.no_grad():
                actions_ = self.network_dict['actor_target'](actor_inputs_)
                critic_inputs_ = T.cat((actor_inputs_, actions_), dim=1)
                value_1_ = self.network_dict['critic_1_target'](critic_inputs_)
                value_2_ = self.network_dict['critic_2_target'](critic_inputs_)
                value_ = T.min(value_1_, value_2_)
                value_target = rewards + done * self.gamma * value_
                value_target = T.clamp(value_target, -self.clip_value, 0.0)

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

            self.actor_optimizer.zero_grad()
            new_actions = self.network_dict['actor'](actor_inputs)
            critic_eval_inputs = T.cat((actor_inputs, new_actions), dim=1).to(self.device)
            new_values_1 = self.network_dict['critic_1'](critic_eval_inputs)
            new_values_2 = self.network_dict['critic_2'](critic_eval_inputs)
            actor_loss = -T.min(new_values_1, new_values_2).mean()
            actor_loss.backward()
            self.actor_optimizer.step()

            critic_losses += critic_loss_1.detach().mean()
            actor_losses += actor_loss.detach().mean()

            self._soft_update(self.network_dict['actor'], self.network_dict['actor_target'])
            self._soft_update(self.network_dict['critic_1'], self.network_dict['critic_1_target'])
            self._soft_update(self.network_dict['critic_2'], self.network_dict['critic_2_target'])

        self.statistic_dict['critic_loss'].append(critic_losses / steps)
        self.statistic_dict['actor_loss'].append(actor_losses / steps)
