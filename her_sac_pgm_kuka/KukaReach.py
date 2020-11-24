import os
import gym
import numpy as np
import pybullet_multigoal_gym as pgm
from plot import smoothed_plot
from collections import namedtuple
from agent.sac_her_continuous import HindsightSACAgent
path = os.path.dirname(os.path.realpath(__file__))

T = namedtuple("transition",
               ('state', 'desired_goal', 'action', 'next_state', 'achieved_goal', 'reward', 'done'))
env = pgm.make("KukaReachSparseEnv-v0")
env.seed(0)
obs = env.reset()
env_params = {'obs_dims': obs['state'].shape[0],
              'goal_dims': obs['desired_goal'].shape[0],
              'action_dims': env.action_space.shape[0],
              'action_max': env.action_space.high,
              'init_input_means': np.zeros((obs['state'].shape[0]+obs['desired_goal'].shape[0],)),
              'init_input_var': np.ones((obs['state'].shape[0]+obs['desired_goal'].shape[0],))
              }
agent = HindsightSACAgent(env_params, T, path=path, seed=300, hindsight=True)
"""
When testing, make sure comment out the mean update(line54), hindsight(line62), and learning(line63)
"""
# Load target networks at epoch 50
# agent.load_network(50)
success_rates = []
cycle_returns = []
EPOCH = 200 + 1
CYCLE = 50
EPISODE = 16

for epo in range(EPOCH):
    for cyc in range(CYCLE):
        ep = 0
        cycle_return = 0
        cycle_timesteps = 0
        while ep < EPISODE:
            done = False
            new_episode = True
            obs = env.reset()
            ep_return = 0
            # start a new episode
            while not done:
                cycle_timesteps += 1
                action = agent.select_action(obs['state'], obs['desired_goal'])
                new_obs, reward, done, info = env.step(action)
                ep_return += reward
                agent.remember(new_episode,
                               obs['state'], obs['desired_goal'], action,
                               new_obs['state'], new_obs['achieved_goal'], reward, 1-int(done))
                agent.normalizer.store_history(np.concatenate((new_obs['state'],
                                                               new_obs['achieved_goal']), axis=0))
                new_episode = False
                obs = new_obs
            agent.normalizer.update_mean()
            ep += 1
            cycle_return += ep_return
            agent.learn()
        cycle_returns.append(cycle_return)
        print("Epoch %i" % epo, "cycle %i" % cyc, "return %i" % cycle_return)

    if (epo % 50 == 0) and (epo != 0):
        agent.save_networks(epo)

smoothed_plot("cycle_returns.png", cycle_returns, x_label="Cycle")
