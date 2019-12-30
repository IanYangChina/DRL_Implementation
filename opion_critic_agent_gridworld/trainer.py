import os
import numpy as np
from plot import smoothed_plot, smoothed_plot_multi_line
from copy import deepcopy as dcp
from collections import namedtuple
from agent.option_critic import OptionCritic


class Trainer(object):
    def __init__(self, env, path, seed=0, hindsight=True,
                 input_rescale=1,
                 training_epoch=201, training_cycle=50, training_episode=16, training_timesteps=70,
                 testing_episode_per_goal=50, testing_timesteps=70, testing_gap=1,
                 saving_gap=50):
        np.set_printoptions(precision=3)
        self.path = path
        OptTr = namedtuple('OptionTransition',
                           ('state', 'inventory', 'desired_goal',
                            'option', 'achieved_goal', 'reward', 'done',
                            'next_state', 'next_inventory', 'next_goal'))

        self.env = env
        _, obs = self.env.reset()
        env_params = {'input_max': self.env.input_max,
                      'input_min': self.env.input_min,
                      'input_rescale': input_rescale,
                      'input_dim': obs['state'].shape[0] + obs['desired_goal_loc'].shape[0] + obs['inventory_vector'].shape[0],
                      'primitive_output_dim': len(self.env.actions),
                      'env_type': self.env.env_type}
        self.hindsight = hindsight
        self.agent = OptionCritic(env_params, OptTr, path=self.path, seed=seed, option_num=4)

        self.training_epoch = training_epoch
        self.training_cycle = training_cycle
        self.training_episode = training_episode
        self.training_timesteps = training_timesteps

        self.testing_episode_per_goal = testing_episode_per_goal
        self.testing_timesteps = testing_timesteps
        self.testing_gap = testing_gap
        self.saving_gap = saving_gap

        self.train_suc_rates = []
        self.test_suc_rates = []
        self.option_policy_loss = None
        self.intra_policy_loss = None
        self.termination_loss = None

    def run(self, opt_optimization_steps, training_render=False, testing_render=False):

        for epo in range(self.training_epoch):
            for cyc in range(self.training_cycle):
                self.train(opt_optimization_steps=opt_optimization_steps, epo=epo, cyc=cyc, render=training_render)

            if epo % self.testing_gap == 0:
                self.test(render=testing_render)
            if (epo % self.saving_gap == 0) and (epo != 0):
                self._save_ckpts(epo)
        self.option_policy_loss = np.array(self.agent.option_policy_loss)
        self.intra_policy_loss = np.array(self.agent.intra_policy_loss)
        self.termination_loss = np.array(self.agent.termination_loss)
        self._plot_success_rates()
        self._save_numpy_to_txt()

    def train(self, opt_optimization_steps, epo=0, cyc=0, render=False):
        sus = 0
        for ep in range(self.training_episode):
            new_episode = True
            ep_time_step = 0
            ep_done = False
            _, obs = self.env.reset()
            while (not ep_done) and (ep_time_step < self.training_timesteps):
                option = self.agent.select_option(obs, ep=((ep+1)*(cyc+1)*(epo+1)*ep_time_step))
                option_termination = False
                while (not option_termination) and (not ep_done) and (ep_time_step < self.training_timesteps):
                    option_termination, action = self.agent.select_action(option, obs)
                    obs_, reward, ep_done = self.env.step(None, obs, action, t=ep_time_step, render=render)
                    ep_time_step += 1
                    sus += reward
                    # store transitions and renew observation
                    self.agent.opt_remember(new_episode,
                                            obs['state'], obs['inventory_vector'], obs['desired_goal_loc'],
                                            option, obs_['achieved_goal_loc'], reward, 1 - int(ep_done),
                                            obs_['state'], obs_['inventory_vector'], obs_['desired_goal_loc'])
                    obs = dcp(obs_)
                    new_episode = False
                    self.agent.intra_policy_learn(obs, obs_, option, action, reward, ep_done)

            self.agent.opt_learn(steps=opt_optimization_steps)
        # save data
        self.train_suc_rates.append(sus / self.training_episode)
        print("Epoch %i" % epo, "Cycle %i" % cyc, "Training success rate {}".format(self.train_suc_rates[-1]))

    def test(self, do_print=False, episode_per_goal=None, render=False):
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
            _, obs = self.env.reset()
            obs['desired_goal'] = self.env.final_goals[goal_ind]
            obs['desired_goal_loc'] = self.env.get_goal_location(obs['desired_goal'])
            while (not ep_done) and (ep_time_step < self.testing_timesteps):
                option = self.agent.select_option(obs, ep=None)
                option_termination = False
                while (not option_termination) and (not ep_done) and (ep_time_step < self.testing_timesteps):
                    option_termination, action = self.agent.select_action(option, obs)
                    obs_, reward, ep_done = self.env.step(None, obs, action, t=ep_time_step, render=render)
                    ep_time_step += 1
                    success[0][goal_ind] += int(reward)
                    if do_print:
                        print("State: {}, option: {}, action: {}, achieved goal: {}".format(obs['state'],
                                                                                            option,
                                                                                            self.env.actions[action],
                                                                                            obs['achieved_goal']))
                    obs = dcp(obs_)
            goal_ind = (goal_ind + 1) % goal_num
        self.test_suc_rates.append(sum(success[0]) / sum(success[1]))
        print("Test success rate {}".format(self.test_suc_rates[-1]))

    def _save_ckpts(self, epo):
        print("Saving check point......")
        self.agent.save_network(epo)

    def _plot_success_rates(self):
        smoothed_plot(self.path + "/Success_rate_train.png", self.train_suc_rates, x_label="Cycle")
        smoothed_plot(self.path + "/Success_rate_test.png", self.test_suc_rates, x_label="Epoch")
        smoothed_plot(self.path + "/option_policy_loss.png", self.option_policy_loss, x_label="Episodes", y_label="Optor mean q value")
        smoothed_plot(self.path + "/intra_policy_loss.png", self.intra_policy_loss, x_label="Timesteps", y_label="Optor mean q value")
        smoothed_plot(self.path + "/termination_loss.png", self.termination_loss, x_label="Timesteps", y_label="Optor mean q value")

    def _save_numpy_to_txt(self):
        path = self.path + "/data"
        if not os.path.isdir(path):
            os.mkdir(path)
        np.savetxt(path+"/tr_opt.dat", np.array(self.train_suc_rates))
        np.savetxt(path+"/te_opt.dat", np.array(self.test_suc_rates))
        np.savetxt(path+"/opt_pi_loss.dat", self.option_policy_loss)
        np.savetxt(path+"/intra_pi_loss.dat", self.intra_policy_loss)
        np.savetxt(path+"/term_loss.dat", self.termination_loss)