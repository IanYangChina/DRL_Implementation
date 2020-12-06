import os
import numpy as np
import gym
import pybullet_envs
from plot import smoothed_plot
from collections import namedtuple
from agent.sac_continuous import SACAgent
path = os.path.dirname(os.path.realpath(__file__))
data_path = path + '/data'
if not os.path.isdir(data_path):
    os.mkdir(data_path)

T = namedtuple("transition",
               ('state', 'action', 'next_state', 'reward', 'done'))
env = gym.make("HalfCheetahBulletEnv-v0")
env.seed(0)
# env.render()
obs = env.reset()
env_params = {'obs_dims': obs.shape[0],
              'action_dims': env.action_space.shape[0],
              'action_max': env.action_space.high,
              'init_input_means': np.zeros((obs.shape[0],)),
              'init_input_var': np.ones((obs.shape[0],))
              }
agent = SACAgent(env_params, T, path=path, seed=300, prioritised=False)
"""
When testing, make sure comment out the mean update(line54), hindsight(line62), and learning(line63)
"""
TEST = False
# Load target networks at epoch 50
if TEST:
    agent.load_network(200)

ep_returns = []
EPISODE = 200

ep = 0
while ep < EPISODE:
    done = False
    new_episode = True
    obs = env.reset()
    ep_return = 0
    # start a new episode
    while not done:
        action = agent.select_action(obs)
        new_obs, reward, done, info = env.step(action)
        ep_return += reward
        agent.remember(obs, action, new_obs, reward, 1-int(done))
        agent.normalizer.store_history(new_obs)
        new_episode = False
        obs = new_obs
        agent.learn(steps=1)
    ep += 1

    ep_returns.append(ep_return)
    print("Episode %i" % ep, "return %0.1f" % ep_return)

    if (ep % 50 == 0) and (ep != 0):
        agent.save_networks(ep)

smoothed_plot(data_path+"/episode_returns.png", ep_returns, x_label="Episode")
