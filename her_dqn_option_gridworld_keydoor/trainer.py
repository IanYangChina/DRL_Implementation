import numpy as np
from plot import smoothed_plot
from copy import deepcopy as dcp
from collections import namedtuple
from agent.herdqn_option_discrete import OptionDQN


class Trainer(object):
    def __init__(self, env, path, is_act_inv=True, torch_seed=0, random_seed=0,
                 training_epoch=401, training_cycle=50, training_episode=16, training_timesteps=70,
                 testing_episode_per_goal=50, testing_timesteps=70, testing_gap=1, saving_gap=50):
        self.path = path
        self.env = env
        opt_obs, act_obs = self.env.reset()
        self.is_act_inv = is_act_inv
        OptTr = namedtuple('OptionTransition',
                           ('state', 'inventory', 'final_goal', 'option', 'next_state', 'next_inventory', 'next_goal',
                            'option_done', 'reward', 'done'))
        ActTr = namedtuple('ActionTransition',
                           ('state', 'inventory', 'desired_goal', 'action', 'next_state', 'next_inventory', 'next_goal',
                            'achieved_goal', 'reward', 'done'))
        env_params = {'input_max': env.input_max,
                      'input_min': env.input_min,
                      'opt_input_dim': opt_obs['state'].shape[0] + opt_obs['final_goal_loc'].shape[0] + opt_obs['inventory_vector'].shape[0],
                      'opt_output_dim': len(env.option_space),
                      'opt_max': np.max(env.option_space),
                      'act_input_dim': act_obs['state'].shape[0] + act_obs['desired_goal_loc'].shape[0] + act_obs['inventory_vector'].shape[0],
                      'act_output_dim': len(env.action_space),
                      'act_max': np.max(env.action_space),
                      'env_type': env.env_type}
        if not self.is_act_inv:
            env_params['act_input_dim'] = act_obs['state'].shape[0] + act_obs['desired_goal_loc'].shape[0]
        self.agent = OptionDQN(env_params, OptTr, ActTr, is_act_inv=is_act_inv, path=self.path,
                               opt_eps_decay_start=training_epoch*training_cycle*training_episode*0.5,
                               torch_seed=torch_seed, random_seed=random_seed)

        self.training_epoch = training_epoch
        self.training_cycle = training_cycle
        self.training_episode = training_episode
        self.training_timesteps = training_timesteps

        self.testing_episode_per_goal = testing_episode_per_goal
        self.testing_timesteps = testing_timesteps
        self.testing_gap = testing_gap
        self.saving_gap = saving_gap

        self.test_success_rates = []
        self.opt_success_rates = []
        self.act_success_rates = []
        self.act_test_sus_rates = []

    def print_training_info(self):
        print("World type: "+self.env.env_type+", layout:")
        for key in self.env.world:
            print(key, self.env.world[key])
        print("Sub-goals:")
        print(self.env.goal_space)

    def run(self):
        for epo in range(self.training_epoch):
            for cyc in range(self.training_cycle):
                self.train(epo=epo, cyc=cyc)
            if epo % self.testing_gap == 0:
                self.act_test()
                self.opt_test()
            if (epo % self.saving_gap == 0) and (epo != 0):
                self.agent.save_network(epo)
        self._plot_success_rates()

    def train(self, epo=0, cyc=0):
        opt_sus = 0
        act_sus = 0
        opt_ep_steps = 0
        for ep in range(self.training_episode):
            opt_ep_returns = 0
            act_ep_returns = 0
            ep_time_step = 0
            ep_done = False
            opt_obs, act_obs = self.env.reset()
            while (not ep_done) and (ep_time_step < self.training_timesteps):
                opt_done = False
                new_option = True
                opt_ep_steps += 1
                option = self.agent.select_option(opt_obs, ep=((ep + 1) * (cyc + 1) * (epo + 1)))
                act_obs['desired_goal'] = self.env.goal_space[option]
                act_obs['desired_goal_loc'] = self.env.get_goal_location(act_obs['desired_goal'])
                while (not opt_done) and (ep_time_step < self.training_timesteps):
                    ep_time_step += 1
                    action = self.agent.select_action(act_obs, ep=((ep + 1) * (cyc + 1) * (epo + 1)))
                    act_obs_, act_reward, ep_done, opt_obs_, opt_reward, opt_done = self.env.step(opt_obs,
                                                                                                  act_obs,
                                                                                                  action)
                    act_obs['achieved_goal_loc'] = dcp(act_obs_['achieved_goal_loc'])
                    opt_ep_returns += opt_reward
                    act_ep_returns += act_reward
                    # store transitions and renew observation
                    self.agent.remember(new_option, "action",
                                        act_obs['state'], act_obs['inventory_vector'], act_obs['desired_goal_loc'],
                                        action,
                                        act_obs_['state'], act_obs_['inventory_vector'],
                                        act_obs_['desired_goal_loc'],
                                        act_obs['achieved_goal_loc'], act_reward, 1 - int(opt_done))
                    act_obs = act_obs_.copy()
                    self.agent.remember(new_option, "option",
                                        opt_obs['state'], opt_obs['inventory_vector'], opt_obs['final_goal_loc'],
                                        option,
                                        opt_obs_['state'], opt_obs_['inventory_vector'], opt_obs_['final_goal_loc'],
                                        1 - int(opt_done), opt_reward, 1 - int(ep_done))
                    opt_obs = opt_obs_.copy()
                    new_option = False
            opt_sus += opt_ep_returns
            act_sus += act_ep_returns
            self.agent.apply_hindsight(hindsight=True)
            self.agent.learn("action", steps=4)
        self.agent.learn("option", steps=4)
        self.opt_success_rates.append(opt_sus / self.training_episode)
        self.act_success_rates.append(act_sus / opt_ep_steps)
        print("Epoch %i" % epo, "Cycle %i" % cyc,
              "Option SucRate {}/{}".format(int(opt_sus), self.training_episode),
              "Action SucRate {}/{}".format(int(act_sus), opt_ep_steps))

    def act_test(self, do_print=False, episode_per_goal=None):

        """Testing primitive agent"""
        goal_num = len(self.env.goal_space)
        success_t = [self.env.goal_space,
                     [0 for g in range(goal_num)],
                     [0 for g in range(goal_num)]]
        goal_ind_t = 0
        if episode_per_goal is None:
            episode_per_goal = self.testing_episode_per_goal
        episodes = goal_num*episode_per_goal
        for ep_t in range(episodes):
            act_obs_t = self.env.reset(act_test=True)
            act_obs_t['desired_goal'] = self.env.goal_space[goal_ind_t]
            act_obs_t['desired_goal_loc'] = self.env.get_goal_location(act_obs_t['desired_goal'])
            success_t[2][goal_ind_t] += 1
            opt_done_t = False
            ep_time_step_t = 0
            if do_print:
                print("\nNew Episode, desired goal: {}".format(act_obs_t['desired_goal']))
            while (not opt_done_t) and (ep_time_step_t < self.testing_timesteps):
                ep_time_step_t += 1
                action_t = self.agent.select_action(act_obs_t)
                act_obs_t_, act_reward_t, opt_done_t = self.env.step(None, act_obs_t, action_t)
                success_t[1][goal_ind_t] += int(act_reward_t)
                act_obs_t['achieved_goal'] = dcp(act_obs_t_['achieved_goal'])
                act_obs_t['achieved_goal_loc'] = dcp(act_obs_t_['achieved_goal_loc'])
                if do_print:
                    print("State: {}, action: {}, achieved goal: {}".format(act_obs_t['state'],
                                                                            self.env.actions[action_t],
                                                                            act_obs_t['achieved_goal']))
                act_obs_t = act_obs_t_.copy()
            goal_ind_t = (goal_ind_t + 1) % goal_num
        self.act_test_sus_rates.append(sum(success_t[1]) / sum(success_t[2]))
        print("Primitive agent test result:\n", success_t)

    def opt_test(self, do_print=False, episode_per_goal=None):
        """Testing both agents"""
        opt_goal_num = len(self.env.final_goals)
        opt_success_t = [self.env.final_goals,
                         [0 for g in range(opt_goal_num)],
                         [0 for g in range(opt_goal_num)]]
        opt_goal_ind_t = 0
        if episode_per_goal is None:
            episode_per_goal = self.testing_episode_per_goal
        episodes = opt_goal_num * episode_per_goal
        for ep_t in range(episodes):
            ep_done_t = False
            ep_time_step_t = 0
            opt_obs_t, act_obs_t = self.env.reset()
            opt_obs_t['final_goal'] = self.env.final_goals[opt_goal_ind_t]
            opt_obs_t['final_goal_loc'] = self.env.get_goal_location(opt_obs_t['final_goal'])
            opt_success_t[2][opt_goal_ind_t] += 1
            if do_print:
                print("\nNew Episode, ultimate goal: {}".format(opt_obs_t['final_goal']))
            while (not ep_done_t) and (ep_time_step_t < self.testing_timesteps):
                option = self.agent.select_option(opt_obs_t, ep=None)
                act_obs_t['desired_goal'] = self.env.goal_space[option]
                act_obs_t['desired_goal_loc'] = self.env.get_goal_location(act_obs_t['desired_goal'])
                opt_done_t = False
                if do_print:
                    print("Option/subgoal: {}".format(act_obs_t['desired_goal']))
                while (not opt_done_t) and (ep_time_step_t < self.testing_timesteps):
                    ep_time_step_t += 1
                    action_t = self.agent.select_action(act_obs_t, ep=None)
                    act_obs_t_, act_reward_t, ep_done_t, opt_obs_t_, opt_reward_t, opt_done_t = \
                        self.env.step(opt_obs_t, act_obs_t, action_t)
                    opt_success_t[1][opt_goal_ind_t] += int(opt_reward_t)
                    act_obs_t['achieved_goal'] = dcp(act_obs_t_['achieved_goal'])
                    act_obs_t['achieved_goal_loc'] = dcp(act_obs_t_['achieved_goal_loc'])
                    if do_print:
                        print("State: {}, action: {}, achieved goal: {}".format(act_obs_t['state'],
                                                                                self.env.actions[action_t],
                                                                                act_obs_t['achieved_goal']))
                    act_obs_t = act_obs_t_.copy()
                    opt_obs_t = opt_obs_t_.copy()
            opt_goal_ind_t = (opt_goal_ind_t + 1) % opt_goal_num
        self.test_success_rates.append(sum(opt_success_t[1])/sum(opt_success_t[2]))
        print("Option agent test result:\n", opt_success_t)

    def _plot_success_rates(self):
        smoothed_plot(self.path + "/Success_rate_train_option.png", self.opt_success_rates, x_label="Cycle")
        smoothed_plot(self.path + "/Success_rate_train_action.png", self.act_success_rates, x_label="Cycle")
        smoothed_plot(self.path + "/Success_rate_test_option.png", self.test_success_rates, x_label="Epo")
        smoothed_plot(self.path + "/Success_rate_test_action.png", self.act_test_sus_rates, x_label="Epo")