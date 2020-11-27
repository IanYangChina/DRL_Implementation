import os
import numpy as np
import pybullet_multigoal_gym as pgm
from plot import smoothed_plot
from collections import namedtuple
from agent.ddpg_her_continuous import HindsightDDPGAgent
path = os.path.dirname(os.path.realpath(__file__))
data_path = path + '/data'
if not os.path.isdir(data_path):
    os.mkdir(data_path)

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
agent = HindsightDDPGAgent(env_params, T, path=path, seed=300, hindsight=True)
"""
When testing, make sure comment out the mean update(line54), hindsight(line62), and learning(line63)
"""
TEST = False
# Load target networks at epoch 50
if TEST:
    agent.load_network(200)
    agent.normalizer.history_mean = np.load(data_path + "/input_means.npy")
    agent.normalizer.history_var = np.load(data_path + "/input_vars.npy")

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
        cycle_successes = 0
        while ep < EPISODE:
            done = False
            new_episode = True
            obs = env.reset()
            ep_return = 0
            # start a new episode
            while not done:
                cycle_timesteps += 1
                action = agent.act(obs['state'], obs['desired_goal'], test=TEST)
                new_obs, reward, done, info = env.step(action)
                ep_return += reward
                agent.remember(new_episode,
                               obs['state'], obs['desired_goal'], action,
                               new_obs['state'], new_obs['achieved_goal'], reward, 1-int(done))
                new_episode = False
                obs = new_obs
            if ep_return > -50:
                cycle_successes += 1
            agent.normalizer.update_mean()
            ep += 1
            cycle_return += ep_return
        success_rate = cycle_successes / EPISODE
        success_rates.append(success_rate)
        cycle_returns.append(cycle_return)
        print("Epoch %i" % epo, "cycle %i" % cyc,
              "return %0.1f" % cycle_return, "success rate %0.2f" % success_rate + "%")
        agent.learn()

    if (epo % 50 == 0) and (epo != 0):
        agent.save_networks(epo)

np.save(data_path + "/input_means", agent.normalizer.history_mean)
np.save(data_path + "/input_vars", agent.normalizer.history_var)
smoothed_plot(data_path+"/success_rates.png", success_rates, x_label="Cycle")
smoothed_plot(data_path+"/cycle_returns.png", cycle_returns, x_label="Cycle")
