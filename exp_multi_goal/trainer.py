import os
import gym
import numpy as np
import pybullet_multigoal_gym as pgm
from plot import smoothed_plot
from collections import namedtuple
T = namedtuple("transition",
               ('state', 'desired_goal', 'action', 'next_state', 'achieved_goal', 'reward', 'done'))


class Trainer(object):
    def __init__(self, env, agent, hindsight, prioritised,
                 seed, path, render=False):
        self.data_path = path + '/data'
        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)
        self.env = pgm.make(env)
        self.env.seed(seed)
        # this argument only works for Mujoco envs, not for PyBullet envs
        self.render = render
        if render:
            self.env.render()
        obs = self.env.reset()
        env_params = {'obs_dims': obs['state'].shape[0],
                      'goal_dims': obs['desired_goal'].shape[0],
                      'action_dims': self.env.action_space.shape[0],
                      'action_max': self.env.action_space.high,
                      'init_input_means': np.zeros((obs['state'].shape[0]+obs['desired_goal'].shape[0],)),
                      'init_input_var': np.ones((obs['state'].shape[0]+obs['desired_goal'].shape[0],))
                      }
        self.agent = agent(env_params, T, path=path, seed=seed, hindsight=hindsight, prioritised=prioritised)

    def run(self, test=False, n_epochs=200, load_network_ep=None):

        if test:
            assert load_network_ep is not None
            self.agent.load_network(load_network_ep)
            self.agent.normalizer.history_mean = np.load(self.data_path + "/input_means.npy")
            self.agent.normalizer.history_var = np.load(self.data_path + "/input_vars.npy")

        success_rates = []
        cycle_returns = []
        EPOCH = n_epochs + 1
        CYCLE = 50
        EPISODE = 16

        for epo in range(EPOCH):
            for cyc in range(CYCLE):
                ep = 0
                cycle_return = 0
                cycle_timesteps = 0
                cycle_successes = 0
                while ep < EPISODE:
                    done = False
                    new_episode = True
                    obs = self.env.reset()
                    ep_return = 0
                    # start a new episode
                    while not done:
                        if self.render:
                            self.env.render()
                        cycle_timesteps += 1
                        action = self.agent.select_action(obs['state'], obs['desired_goal'], test)
                        new_obs, reward, done, info = self.env.step(action)
                        ep_return += reward
                        if not test:
                            self.agent.remember(new_episode,
                                           obs['state'], obs['desired_goal'], action,
                                           new_obs['state'], new_obs['achieved_goal'], reward, 1-int(done))
                            self.agent.normalizer.store_history(np.concatenate((new_obs['state'],
                                                                           new_obs['desired_goal']), axis=0))
                        new_episode = False
                        obs = new_obs
                    if ep_return > -50:
                        cycle_successes += 1
                    if not test:
                        self.agent.normalizer.update_mean()
                    ep += 1
                    cycle_return += ep_return
                success_rate = cycle_successes / EPISODE
                success_rates.append(success_rate)
                cycle_returns.append(cycle_return)
                print("Epoch %i" % epo, "cycle %i" % cyc,
                      "return %0.1f" % cycle_return, "success rate %0.2f" % success_rate + "%")
                if not test:
                    self.agent.learn()

            if (epo % 50 == 0) and (epo != 0) and (not test):
                self.agent.save_networks(epo)

        if not test:
            np.save(self.data_path + "/input_means", self.agent.normalizer.history_mean)
            np.save(self.data_path + "/input_vars", self.agent.normalizer.history_var)
            smoothed_plot(self.data_path+"/success_rates.png", success_rates, x_label="Cycle")
            smoothed_plot(self.data_path+"/cycle_returns.png", cycle_returns, x_label="Cycle")
