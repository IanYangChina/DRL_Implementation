import os
import numpy as np
import pybullet_envs
from agent import agents
from plot import smoothed_plot


class Trainer(object):
    def __init__(self, env, seed, render, path, agent_type, prioritised=False, discard_time_limit=False, update_interval=1):
        if not os.path.isdir(path):
            os.mkdir(path)
        self.data_path = path + '/data'
        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)
        self.env = pybullet_envs.make(env)
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
        assert agent_type in ['ddpg', 'sac', 'ppo']
        self.agent_type = agent_type
        self.agent = agents[agent_type](env_params, path=path, seed=seed, prioritised=prioritised, discard_time_limit=discard_time_limit)
        self.update_interval = update_interval

    def run(self, test=False, n_episodes=101, load_network_ep=None):
        if test:
            assert load_network_ep is not None
            self.agent.load_network(load_network_ep)
            self.agent.normalizer.history_mean = np.load(self.data_path + "/input_means.npy")
            self.agent.normalizer.history_var = np.load(self.data_path + "/input_vars.npy")

        ep_returns = []
        EPISODE = n_episodes

        ep = 0
        step = 0
        while ep < EPISODE:
            done = False
            obs = self.env.reset()
            ep_return = 0
            # start a new episode
            while not done:
                if self.agent_type == 'ppo':
                    action, prob = self.agent.select_action(obs, log_probs=True, test=test)
                else:
                    action = self.agent.select_action(obs, test=test)
                new_obs, reward, done, info = self.env.step(action)
                ep_return += reward
                if not test:
                    if self.agent_type == 'ppo':
                        self.agent.remember(obs, action, prob, new_obs, reward, 1-int(done))
                    else:
                        self.agent.remember(obs, action, new_obs, reward, 1-int(done))
                    self.agent.normalizer.store_history(new_obs)
                    self.agent.normalizer.update_mean()
                    if (step % self.update_interval == 0) and (step != 0):
                        self.agent.learn()
                obs = new_obs
                step += 1
            ep += 1

            ep_returns.append(ep_return)
            print("Episode %i" % ep, "return %0.1f" % ep_return)

            if (ep % 25 == 0) and (ep != 0) and (not test):
                self.agent.save_networks(ep)

        if not test:
            # smoothed_plot(self.data_path+"/alpha.png", self.agent.alpha_record, x_label='Timestep', y_label='Alpha', window=5)
            # np.save(self.data_path + '/alpha_record', self.agent.alpha_record)
            # smoothed_plot(self.data_path+"/policy_entropy.png", self.agent.policy_entropy_record, x_label='Timestep', y_label='Policy entropy', window=5)
            # np.save(self.data_path + '/policy_entropy_record', self.agent.policy_entropy_record)
            smoothed_plot(self.data_path+"/episode_returns.png", ep_returns, x_label="Episode")
            np.save(self.data_path + '/episode_returns', ep_returns)

            np.save(self.data_path + "/input_means", self.agent.normalizer.history_mean)
            np.save(self.data_path + "/input_vars", self.agent.normalizer.history_var)
            return ep_returns
        else:
            return print("Test finished")
