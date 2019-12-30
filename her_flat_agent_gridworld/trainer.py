import os
import numpy as np
from plot import smoothed_plot
from copy import deepcopy as dcp
from collections import namedtuple
from agent.her_dqn_discrete import HindsightDQN
from agent.her_td3_discrete import HindsightTD3


class Trainer(object):
    def __init__(self, env, path, is_inv=True, seed=0, agent_type="dqn",
                 training_epoch=101, training_cycle=50, training_episode=16, training_timesteps=70,
                 testing_episode_per_goal=50, testing_timesteps=70, testing_gap=1, saving_gap=10):
        self.path = path+"/"+agent_type
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        self.env = env
        opt_obs, obs = self.env.reset()
        self.is_inv = is_inv
        ActTr = namedtuple('ActionTransition',
                           ('state', 'inventory', 'desired_goal', 'action', 'next_state', 'next_inventory', 'next_goal',
                            'achieved_goal', 'reward', 'done'))
        env_params = {'input_max': env.input_max,
                      'input_min': env.input_min,
                      'input_dim': obs['state'].shape[0] + obs['desired_goal_loc'].shape[0] + obs['inventory_vector'].shape[0],
                      'output_dim': len(env.action_space),
                      'max': np.max(env.action_space),
                      'env_type': env.env_type}
        if not self.is_inv:
            env_params['input_dim'] = obs['state'].shape[0] + obs['desired_goal_loc'].shape[0]

        if agent_type == "dqn":
            self.agent = HindsightDQN(env_params, ActTr, path=self.path, seed=seed, hindsight=True)
        elif agent_type == "td3":
            self.agent = HindsightTD3(env_params, ActTr, path=self.path, seed=seed, hindsight=True)
        else:
            raise ValueError("Agent: {} doesn't exist, choose one among ['dqn', 'td3'], "
                             "default type is 'dqn".format(agent_type))

        self.training_epoch = training_epoch
        self.training_cycle = training_cycle
        self.training_episode = training_episode
        self.training_timesteps = training_timesteps

        self.testing_episode_per_goal = testing_episode_per_goal
        self.testing_timesteps = testing_timesteps
        self.testing_gap = testing_gap
        self.saving_gap = saving_gap

        self.test_success_rates = []
        self.success_rates = []
        self.test_sus_rates = []

    def print_training_info(self):
        print("World type: "+self.env.env_type+", layout:")
        for key in self.env.world:
            print(key, self.env.world[key])
        print("Sub-goals:")
        print(self.env.goal_space)

    def run(self, training_render=False, testing_render=False):
        for epo in range(self.training_epoch):
            for cyc in range(self.training_cycle):
                self.train(epo=epo, cyc=cyc, render=training_render)
            if epo % self.testing_gap == 0:
                self.test(render=testing_render)
            if (epo % self.saving_gap == 0) and (epo != 0):
                self.agent.save_network(epo)
        self._plot_success_rates()

    def train(self, epo=0, cyc=0, render=False):
        sus = 0
        for ep in range(self.training_episode):
            new_episode = True
            ep_returns = 0
            ep_time_step = 0
            ep_done = False
            obs = self.env.reset(act_test=True)
            while (not ep_done) and (ep_time_step < self.training_timesteps):
                action = self.agent.select_action(obs, ep=((ep + 1) * (cyc + 1) * (epo + 1)))
                obs_, reward, ep_done = self.env.step(None, obs, action, t=ep_time_step, render=render)
                ep_time_step += 1
                obs['achieved_goal_loc'] = dcp(obs_['achieved_goal_loc'])
                ep_returns += reward
                # store transitions and renew observation
                self.agent.remember(new_episode,
                                    obs['state'], obs['inventory_vector'], obs['desired_goal_loc'],
                                    action,
                                    obs_['state'], obs_['inventory_vector'],
                                    obs_['desired_goal_loc'],
                                    obs['achieved_goal_loc'], reward, 1 - int(ep_done))
                obs = obs_.copy()
                new_episode = False
            sus += ep_returns
            self.agent.learn()
        self.success_rates.append(sus / self.training_episode)
        print("Epoch %i" % epo, "Cycle %i" % cyc, "SucRate {}/{}".format(int(sus), self.training_episode))

    def test(self, do_print=False, episode_per_goal=None, render=False):

        """Testing primitive agent"""
        goal_num = len(self.env.goal_space)
        success_t = [self.env.goal_space,
                     [0 for _ in range(goal_num)],
                     [0 for _ in range(goal_num)]]
        goal_ind_t = 0
        if episode_per_goal is None:
            episode_per_goal = self.testing_episode_per_goal
        episodes = goal_num*episode_per_goal
        for ep_t in range(episodes):
            obs_t = self.env.reset(act_test=True)
            obs_t['desired_goal'] = self.env.goal_space[goal_ind_t]
            obs_t['desired_goal_loc'] = self.env.get_goal_location(obs_t['desired_goal'])
            success_t[2][goal_ind_t] += 1
            ep_done_t = False
            ep_time_step_t = 0
            if do_print:
                print("\nNew Episode, desired goal: {}".format(obs_t['desired_goal']))
            while (not ep_done_t) and (ep_time_step_t < self.testing_timesteps):
                action_t = self.agent.select_action(obs_t)
                obs_t_, reward_t, ep_done_t = self.env.step(None, obs_t, action_t, t=ep_time_step_t, render=render)
                ep_time_step_t += 1
                success_t[1][goal_ind_t] += int(reward_t)
                obs_t['achieved_goal'] = dcp(obs_t_['achieved_goal'])
                obs_t['achieved_goal_loc'] = dcp(obs_t_['achieved_goal_loc'])
                if do_print:
                    print("State: {}, action: {}, achieved goal: {}".format(obs_t['state'],
                                                                            self.env.actions[action_t],
                                                                            obs_t['achieved_goal']))
                obs_t = obs_t_.copy()
            goal_ind_t = (goal_ind_t + 1) % goal_num
        self.test_sus_rates.append(sum(success_t[1]) / sum(success_t[2]))
        print("Primitive agent test result:\n", success_t)

    def _plot_success_rates(self):
        smoothed_plot(self.path + "/Success_rate_train_action.png", self.success_rates, x_label="Cycle")
        smoothed_plot(self.path + "/Success_rate_test_action.png", self.test_sus_rates, x_label="Epo")