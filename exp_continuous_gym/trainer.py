import os
import numpy as np
import gym
import pybullet_envs
from plot import smoothed_plot
from collections import namedtuple


T = namedtuple("transition",
               ('state', 'action', 'next_state', 'reward', 'done'))


class Trainer(object):
    def __init__(self, env, agent, prioritised, seed, render, path):
        self.data_path = path + '/data'
        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)
        self.env = gym.make(env)
        self.env.seed(seed)
        if render:
            self.env.render()
        obs = self.env.reset()
        env_params = {'obs_dims': obs.shape[0],
                      'action_dims': self.env.action_space.shape[0],
                      'action_max': self.env.action_space.high,
                      'init_input_means': np.zeros((obs.shape[0],)),
                      'init_input_var': np.ones((obs.shape[0],))
                      }
        self.agent = agent(env_params, T, path=path, seed=seed, prioritised=prioritised)

    def run(self, test=False, n_episodes=100, load_network_ep=None):
        if test:
            assert load_network_ep is not None
            self.agent.load_network(load_network_ep)
            self.agent.normalizer.history_mean = np.load(self.data_path + "/input_means.npy")
            self.agent.normalizer.history_var = np.load(self.data_path + "/input_vars.npy")

        ep_returns = []
        EPISODE = n_episodes

        ep = 0
        while ep < EPISODE:
            done = False
            obs = self.env.reset()
            ep_return = 0
            # start a new episode
            while not done:
                action = self.agent.select_action(obs, test)
                new_obs, reward, done, info = self.env.step(action)
                ep_return += reward
                if not test:
                    self.agent.remember(obs, action, new_obs, reward, 1-int(done))
                    self.agent.normalizer.store_history(new_obs)
                    self.agent.normalizer.update_mean()
                    self.agent.learn(steps=1)
                obs = new_obs
            ep += 1

            ep_returns.append(ep_return)
            print("Episode %i" % ep, "return %0.1f" % ep_return)

            if (ep % 50 == 0) and (ep != 0) and (not test):
                self.agent.save_networks(ep)

        if not test:
            smoothed_plot(self.data_path+"/episode_returns.png", ep_returns, x_label="Episode")
            np.save(self.data_path + "/input_means", self.agent.normalizer.history_mean)
            np.save(self.data_path + "/input_vars", self.agent.normalizer.history_var)

