import os
import gym
import numpy as np
from copy import deepcopy as dcp
from scipy.misc import imresize  # deprecated after scipy 1.3.0
from agent.dqn_atari import DQN
from plot import smoothed_plot


class Trainer(object):
    def __init__(self, path, env='Breakout-v0', seed=0, training_episode=2001, training_timesteps=200, saving_gap=10,
                 frame_skip=4,
                 main_net_update_frequency=4, target_net_update_frequency=10000):
        self.path = path + "/results"
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        self.env = gym.make(env)
        obs = self.env.reset()
        env_params = {'action_num': self.env.action_space.n}
        self.image_height = obs.shape[0]
        self.agent = DQN(env_params, path=self.path, seed=seed)
        self.state = []
        self.frame_skip = frame_skip
        self.main_net_update_frequency = main_net_update_frequency
        self.target_net_update_frequency = target_net_update_frequency

        self.training_episode = training_episode
        self.training_timesteps = training_timesteps
        self.saving_gap = saving_gap

        self.training_return = []

    def warm_start(self, n=50000):
        print("Running warm-start...")
        while len(self.agent.memory) < n:
            self.env.reset()
            frame, _, _, lives = self.env.step(1)
            for i in range(30):
                frame, _, _, _ = self.env.step(0)

            num_lives = lives['ale.lives']
            obs = self.pre_process(frame)
            done = False
            while not done:
                action = self.env.action_space.sample()
                reward = 0
                for fs in range(self.frame_skip):
                    frame, r, done, lives = self.env.step(action)
                    reward += r
                    if done:
                        reward = -1
                        break
                if num_lives > lives['ale.lives']:
                    reward = -1
                    num_lives = lives['ale.lives']
                if reward > 0:
                    reward = 1
                elif reward < 0:
                    reward = -1
                obs_ = self.pre_process(frame)
                self.agent.memory.store_experience(obs, action, obs_, reward, done)
                obs = obs_.copy()
        print("Warm-start finished")

    def train(self, render=False):
        print("Start training...")
        ts_count = 0
        for ep in range(self.training_episode):
            self.env.reset()
            frame, _, _, lives = self.env.step(1)
            for i in range(30):
                frame, _, _, _ = self.env.step(0)
            num_lives = lives['ale.lives']
            obs = self.pre_process(frame)
            done = False
            ep_return = 0
            for t in range(self.training_timesteps):
                if render:
                    self.env.render()
                action = self.agent.select_action(obs, ts_count=ts_count)

                reward = 0
                for fs in range(self.frame_skip):
                    frame, r, done, lives = self.env.step(action)
                    reward += r
                    if done:
                        reward = -1
                        break
                ts_count += 1
                if num_lives > lives['ale.lives']:
                    reward = -1
                    num_lives = lives['ale.lives']
                if reward > 0:
                    reward = 1
                elif reward < 0:
                    reward = -1
                obs_ = self.pre_process(frame)
                ep_return += reward
                self.agent.memory.store_experience(obs, action, obs_, reward, done)
                if ts_count % self.main_net_update_frequency == 0:
                    self.agent.learn(steps=1)
                if ts_count % self.target_net_update_frequency == 0:
                    self.agent.soft_update(tau=1.0)
                obs = dcp(obs_)
                if done:
                    # print("Episode finishes after {} timesteps".format(t+1))
                    break

            self.training_return.append(ep_return)
            print("Episode {}, return {}".format(ep, ep_return))
            if (ep % self.saving_gap == 0) or (ep == self.training_episode-1):
                self.agent.save_network(ep)

        smoothed_plot("ep_returns.png", self.training_return, x_label="Episodes", window=20)

    def pre_process(self, frame):
        # https://github.com/R-Stefano/DQN/blob/8c4d5634453bbb130ecf4624fff09987b107ec84/DQN.py#L207
        # Convert RGB to grayscale [210,160,3] -> [210,160,1]
        gray_frame = np.mean(frame, axis=2)
        # Downsampling the image [210,160,1] -> [110,84,1]
        downsampled_frame = imresize(gray_frame, [110, 84])
        # try and error to set the right height crop for the image in order that it is 84x84
        gap = 26
        processed_frame = (downsampled_frame[gap:self.image_height + gap][:]).astype(np.uint8)
        if len(self.state) == 4:
            del self.state[0]
            self.state.append(processed_frame)
        else:
            while len(self.state) != 4:
                self.state.append(processed_frame)
        # return the state [4,84,84] as [84,84,4]
        state = np.array(self.state)
        return state
        # return np.transpose(self.state, (1, 2, 0))