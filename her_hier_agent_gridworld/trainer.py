import os
import numpy as np
from plot import smoothed_plot, smoothed_plot_multi_line
from copy import deepcopy as dcp
from collections import namedtuple
from agent.herdqn_option_discrete import OptionDQN
from agent.utils.demonstrator import Demonstrator


class Trainer(object):
    def __init__(self, env, path, seed=0,
                 demonstrations=None, use_demonstrator_in_training=False, demonstrated_episodes=2,
                 act_exploration=None, sub_suc_percentage=None, gsrb_decay=None, act_eps_decay=30000,
                 act_hindsight=True,
                 input_rescale=1,
                 training_epoch=201, training_cycle=50, training_episode=16, training_timesteps=70,
                 testing_episode_per_goal=50, testing_timesteps=70, testing_gap=1,
                 saving_gap=50):
        np.set_printoptions(precision=3)
        self.path = path
        OptTr = namedtuple('OptionTransition',
                           ('state', 'inventory', 'final_goal', 'option', 'next_state', 'next_inventory', 'next_goal',
                            'option_done', 'reward', 'done'))
        ActTr = namedtuple('ActionTransition',
                           ('state', 'inventory', 'desired_goal', 'action', 'next_state', 'next_inventory', 'next_goal',
                            'achieved_goal', 'reward', 'done'))

        self.env = env
        opt_obs, act_obs = self.env.reset()
        env_params = {'input_max': self.env.input_max,
                      'input_min': self.env.input_min,
                      'input_rescale': input_rescale,
                      'opt_input_dim': opt_obs['state'].shape[0] + opt_obs['final_goal_loc'].shape[0] + opt_obs['inventory_vector'].shape[0],
                      'opt_output_dim': len(self.env.option_space),
                      'opt_max': np.max(self.env.option_space),
                      'goals': self.env.goal_space,
                      'act_input_dim': act_obs['state'].shape[0] + act_obs['desired_goal_loc'].shape[0] + act_obs['inventory_vector'].shape[0],
                      'act_output_dim': len(self.env.action_space),
                      'act_max': np.max(self.env.action_space),
                      'env_type': self.env.env_type}
        self.act_exploration = act_exploration
        self.act_hindsight = act_hindsight
        self.agent = OptionDQN(env_params, OptTr, ActTr,
                               path=self.path, seed=seed,
                               opt_eps_decay_start=training_epoch*training_cycle*training_episode*0.5,
                               act_exploration=self.act_exploration, sub_suc_percentage=sub_suc_percentage,
                               gsrb_decay=gsrb_decay, act_eps_decay=act_eps_decay)

        if demonstrations is None:
            """Example sequences of indexes of sub-goals: 
               demonstrations = [[0, 1, 2, 3, 4], [1, 0, 2, 3, 4]]
            """
            raise ValueError("Demonstrations need to be specify either for testing or demonstrating")
        self.demonstrations = demonstrations
        self.demonstrator = Demonstrator(demonstrations)
        self.use_demonstrator_in_training = use_demonstrator_in_training
        self.demonstrated_episodes = demonstrated_episodes

        self.training_epoch = training_epoch
        self.training_cycle = training_cycle
        self.training_episode = training_episode
        self.training_timesteps = training_timesteps

        self.testing_episode_per_goal = testing_episode_per_goal
        self.testing_timesteps = testing_timesteps
        self.testing_gap = testing_gap
        self.saving_gap = saving_gap

        self.opt_train_goal_counts = []
        self.opt_train_suc_rates = []
        self.opt_test_suc_rates = []
        self.opt_mean_q = None
        self.act_train_suc_rates = []
        self.act_test_suc_rates = []
        self.act_test_suc_rates_per_goal = []
        self.act_train_suc_rates_per_goal_ep = np.zeros(len(self.env.goal_space))
        self.act_train_suc_rates_per_goal = []
        self.act_epsilons = []
        self.act_mean_q = None
        self.dem_test_suc_rates = []

    def run(self, opt_optimization_steps, act_optimization_steps):

        for epo in range(self.training_epoch):
            self.opt_train_goal_counts.append(np.zeros(len(self.env.goal_space)))
            self.act_train_suc_rates_per_goal_ep = np.zeros(len(self.env.goal_space))
            for cyc in range(self.training_cycle):
                self.train(opt_optimization_steps=opt_optimization_steps, act_optimization_steps=act_optimization_steps,
                           epo=epo, cyc=cyc)
                self.agent.opt_mean_q.append(np.mean(self.agent.opt_mean_q_tmp))
                self.agent.act_mean_q.append(np.mean(self.agent.act_mean_q_tmp))

            self.act_train_suc_rates_per_goal_ep /= self.training_cycle
            self.act_train_suc_rates_per_goal.append(dcp(self.act_train_suc_rates_per_goal_ep))
            if epo % self.testing_gap == 0:
                print("Epoch %i" % epo)
                print("Act train success rate per goal: ", self.act_train_suc_rates_per_goal[-1])
                self.opt_test()
                self.act_test()
                print("Act test success rate per goal: ", self.act_test_suc_rates_per_goal[-1])
                if self.act_exploration is "GSRB":
                    self.agent.act_exploration.update_epsilons(self.act_test_suc_rates_per_goal[-1])
                    self.agent.act_exploration.print_epsilons(epo*self.training_cycle*self.training_episode)
                    self.act_epsilons.append(dcp(self.agent.act_exploration.epsilons_epo))
            if (epo % self.saving_gap == 0) and (epo != 0):
                self._save_ckpts(epo)
        self.opt_mean_q = np.array(self.agent.opt_mean_q)
        self.act_mean_q = np.array(self.agent.act_mean_q)
        self._plot_success_rates()
        self._save_numpy_to_txt()

    def train(self, opt_optimization_steps, act_optimization_steps, epo=0, cyc=0):
        success = [[0 for _ in range(len(self.env.goal_space))],
                   [0 for _ in range(len(self.env.goal_space))]]
        opt_sus = 0
        act_sus = 0
        for ep in range(self.training_episode):
            demon = False
            if self.use_demonstrator_in_training and ((ep+self.demonstrated_episodes) >= self.training_episode):
                demon = True
            opt_ep_returns = 0
            act_ep_returns = 0
            ep_time_step = 0
            ep_done = False
            opt_obs, act_obs = self.env.reset()
            if demon:
                self.demonstrator.reset()
                demon_goal_ind = self.demonstrator.get_final_goal()
                opt_obs['final_goal'] = self.env.goal_space[demon_goal_ind]
                opt_obs['final_goal_loc'] = self.env.get_goal_location(opt_obs['final_goal'])
            while (not ep_done) and (ep_time_step < self.training_timesteps):
                opt_done = False
                new_option = True
                if demon:
                    option = self.demonstrator.get_next_goal()
                else:
                    option = self.agent.select_option(opt_obs, ep=((ep + 1) * (cyc + 1) * (epo + 1)))
                success[1][option] += 1
                act_obs['desired_goal'] = self.env.goal_space[option]
                act_obs['desired_goal_loc'] = self.env.get_goal_location(act_obs['desired_goal'])
                while (not opt_done) and (not ep_done) and (ep_time_step < self.training_timesteps):
                    ep_time_step += 1
                    action = self.agent.select_action(act_obs, ep=((ep + 1) * (cyc + 1) * (epo + 1)))
                    act_obs_, act_reward, ep_done, opt_obs_, opt_reward, opt_done = self.env.step(opt_obs,
                                                                                                  act_obs,
                                                                                                  action)
                    opt_ep_returns += opt_reward
                    act_ep_returns += act_reward
                    success[0][option] += act_reward
                    # store transitions and renew observation
                    self.agent.remember(new_option, "action",
                                        act_obs['state'], act_obs['inventory_vector'], act_obs['desired_goal_loc'],
                                        action,
                                        act_obs_['state'], act_obs_['inventory_vector'], act_obs_['desired_goal_loc'],
                                        act_obs_['achieved_goal_loc'],
                                        act_reward, 1 - int(opt_done))
                    act_obs = dcp(act_obs_)
                    self.agent.remember(new_option, "option",
                                        opt_obs['state'], opt_obs['inventory_vector'], opt_obs['final_goal_loc'],
                                        option,
                                        opt_obs_['state'], opt_obs_['inventory_vector'], opt_obs_['final_goal_loc'],
                                        1 - int(opt_done), opt_reward, 1 - int(ep_done))
                    opt_obs = dcp(opt_obs_)
                    new_option = False
            opt_sus += opt_ep_returns
            act_sus += act_ep_returns
            self.agent.apply_hindsight(hindsight=self.act_hindsight)
            self.agent.learn("action", steps=act_optimization_steps)
            self.agent.learn("option", steps=opt_optimization_steps)
        # save data
        self.opt_train_suc_rates.append(opt_sus / self.training_episode)
        self.act_train_suc_rates.append(act_sus / sum(success[1]))
        self.opt_train_goal_counts[-1] += np.array(success[1])
        self.act_train_suc_rates_per_goal_ep += np.array(success[0])/np.array(success[1])
        nan_ind = np.argwhere(np.isnan(self.act_train_suc_rates_per_goal_ep))
        if nan_ind.shape[0] > 0:
            for ind in range(nan_ind.shape[0]):
                self.act_train_suc_rates_per_goal_ep[nan_ind[ind][0]] = 0

    def act_test(self, do_print=False, episode_per_demon=None):
        demon_ind = 0
        demon_num = len(self.demonstrations)
        if episode_per_demon is None:
            episode_per_demon = self.testing_episode_per_goal//2
        episodes = demon_num*episode_per_demon
        dem_success = [[0 for _ in range(demon_num)],
                       [episode_per_demon for _ in range(demon_num)]]
        goal_num = len(self.env.goal_space)
        success = [[0 for _ in range(goal_num)],
                   [episode_per_demon*sum(x.count(n) for x in self.demonstrations) for n in range(goal_num)]]

        for ep in range(episodes):
            dem_done = False
            ep_time_step = 0
            act_obs = self.env.reset(act_test=True)
            dem_return = 0
            self.demonstrator.manual_reset(demon_ind=demon_ind)
            while (not dem_done) and (ep_time_step < self.testing_timesteps):
                goal_ind = self.demonstrator.get_next_goal()
                act_obs['desired_goal'] = self.env.goal_space[goal_ind]
                act_obs['desired_goal_loc'] = self.env.get_goal_location(act_obs['desired_goal'])
                goal_done = False
                if do_print:
                    print("Option/subgoal: {}".format(act_obs['desired_goal']))
                while (not goal_done) and (not dem_done) and (ep_time_step < self.testing_timesteps):
                    ep_time_step += 1
                    action = self.agent.select_action(act_obs, ep=None)
                    act_obs_, act_reward, goal_done = self.env.step(None, act_obs, action)
                    success[0][goal_ind] += act_reward
                    dem_return += int(act_reward)
                    if do_print:
                        print("State: {}, action: {}, achieved goal: {}".format(act_obs['state'],
                                                                                self.env.actions[action],
                                                                                act_obs_['achieved_goal']))
                    act_obs = dcp(act_obs_)
                if goal_ind == self.demonstrator.current_final_goal:
                    dem_done = True

            if dem_return == len(self.demonstrations[demon_ind]):
                dem_success[0][demon_ind] += 1
            demon_ind = (demon_ind + 1) % demon_num
        self.act_test_suc_rates_per_goal.append(np.array(success[0])/np.array(success[1]))
        self.act_test_suc_rates.append(sum(success[0])/sum(success[1]))
        self.dem_test_suc_rates.append(sum(dem_success[0])/sum(dem_success[1]))
        print("Action policy test result:", success)
        print("Demonstration test result:", dem_success)

    def opt_test(self, do_print=False, episode_per_goal=None):
        """Testing both agents"""
        if episode_per_goal is None:
            episode_per_goal = self.testing_episode_per_goal
        goal_ind = 0
        goal_num = len(self.env.final_goals)
        success = [[0 for _ in range(goal_num)],
                   [episode_per_goal for _ in range(goal_num)]]
        episodes = goal_num * episode_per_goal
        for ep in range(episodes):
            ep_done = False
            ep_time_step = 0
            opt_obs, act_obs = self.env.reset()
            opt_obs['final_goal'] = self.env.final_goals[goal_ind]
            opt_obs['final_goal_loc'] = self.env.get_goal_location(opt_obs['final_goal'])
            if do_print:
                print("\nNew Episode, ultimate goal: {}".format(opt_obs['final_goal']))
            while (not ep_done) and (ep_time_step < self.testing_timesteps):
                option = self.agent.select_option(opt_obs, ep=None)
                act_obs['desired_goal'] = self.env.goal_space[option]
                act_obs['desired_goal_loc'] = self.env.get_goal_location(act_obs['desired_goal'])
                opt_done = False
                if do_print:
                    print("Option/subgoal: {}".format(act_obs['desired_goal']))
                while (not opt_done) and (not ep_done) and (ep_time_step < self.testing_timesteps):
                    ep_time_step += 1
                    action = self.agent.select_action(act_obs, ep=None)
                    act_obs_, act_reward, ep_done, opt_obs_, opt_reward, opt_done = \
                        self.env.step(opt_obs, act_obs, action)
                    success[0][goal_ind] += int(opt_reward)
                    if do_print:
                        print("State: {}, action: {}, achieved goal: {}".format(act_obs['state'],
                                                                                self.env.actions[action],
                                                                                act_obs_['achieved_goal']))
                    act_obs = dcp(act_obs_)
                    opt_obs = dcp(opt_obs_)
            goal_ind = (goal_ind + 1) % goal_num
        self.opt_test_suc_rates.append(sum(success[0]) / sum(success[1]))
        print("Option agent test result: ", success)

    def _save_ckpts(self, epo):
        print("Saving check point......")
        self.agent.save_network(epo)

    def _plot_success_rates(self):
        smoothed_plot(self.path + "/Success_rate_train_option.png", self.opt_train_suc_rates, x_label="Cycle")
        smoothed_plot(self.path + "/Success_rate_train_action.png", self.act_train_suc_rates, x_label="Cycle")
        smoothed_plot(self.path + "/Success_rate_test_option.png", self.opt_test_suc_rates, x_label="Epoch")
        smoothed_plot(self.path + "/Success_rate_test_action.png", self.act_test_suc_rates, x_label="Epoch")
        smoothed_plot(self.path + "/Optor_mean_q.png", self.opt_mean_q, x_label="Cycle", y_label="Optor mean q value")
        smoothed_plot(self.path + "/Actor_mean_q.png", self.act_mean_q, x_label="Cycle", y_label="Optor mean q value")
        if self.use_demonstrator_in_training:
            smoothed_plot(self.path + "/Success_rate_test_demon.png", self.dem_test_suc_rates, x_label="Epoch")
        self._plot_success_rates_per_goal()

    def _plot_success_rates_per_goal(self):
        legend = dcp(self.env.goal_space)
        train_suc_rates = np.array(self.act_train_suc_rates_per_goal)
        train_suc_rates = np.transpose(train_suc_rates)
        test_suc_rates = np.array(self.act_test_suc_rates_per_goal)
        test_suc_rates = np.transpose(test_suc_rates)
        train_goal_counts = np.array(self.opt_train_goal_counts)
        train_goal_counts = np.transpose(train_goal_counts)
        assert train_suc_rates.shape[0] == test_suc_rates.shape[0] == train_goal_counts.shape[0]
        smoothed_plot_multi_line(self.path+"/Success_rate_test_goal.png", test_suc_rates,
                                 legend=legend, x_label="Epoch")
        smoothed_plot_multi_line(self.path+"/Success_rate_train_goal.png", train_suc_rates,
                                 legend=legend, x_label="Epoch")
        smoothed_plot_multi_line(self.path+"/Counts_train_goal.png", train_goal_counts,
                                 legend=legend, x_label="Epoch", y_label="Subgoal counts")

        if self.act_exploration is not None:
            epsilons = np.array(self.act_epsilons)
            epsilons = np.transpose(epsilons)
            assert train_suc_rates.shape[0] == test_suc_rates.shape[0] == epsilons.shape[0]
            assert len(self.act_epsilons) > 2
            smoothed_plot_multi_line(self.path + "/Epsilons_goal.png", epsilons,
                                     legend=legend, x_label="Epoch", y_label="Epsilons")

    def _save_numpy_to_txt(self):
        path = self.path + "/data"
        if not os.path.isdir(path):
            os.mkdir(path)
        tr_opt = np.array(self.opt_train_suc_rates)
        np.savetxt(path+"/tr_opt.dat", tr_opt)
        tr_act = np.array(self.act_train_suc_rates)
        np.savetxt(path+"/tr_act.dat", tr_act)
        te_opt = np.array(self.opt_test_suc_rates)
        np.savetxt(path+"/te_opt.dat", te_opt)
        te_act = np.array(self.act_test_suc_rates)
        np.savetxt(path+"/te_act.dat", te_act)
        tr_goal = np.array(self.act_train_suc_rates_per_goal)
        np.savetxt(path+"/tr_goal.dat", tr_goal)
        te_goal = np.array(self.act_test_suc_rates_per_goal)
        np.savetxt(path+"/te_goal.dat", te_goal)
        tr_goal_counts = np.array(self.opt_train_goal_counts)
        np.savetxt(path+"/tr_goal_counts.dat", tr_goal_counts)
        np.savetxt(path+"/opt_mean_q.dat", self.opt_mean_q)
        np.savetxt(path+"/act_mean_q.dat", self.act_mean_q)
        if self.act_exploration is not None:
            eps_goal = np.array(self.act_epsilons)
            np.savetxt(path + "/eps_goal.dat", eps_goal)
        if self.use_demonstrator_in_training:
            dem = np.array(self.dem_test_suc_rates)
            np.savetxt(path+"/dem.dat", dem)