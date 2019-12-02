import os
import gym
import numpy as np
from plot import smoothed_plot
from collections import namedtuple
from agent.herddpg_continuous import DDPGAgent
path = os.path.dirname(os.path.realpath(__file__))

T = namedtuple("transition",
               ('state', 'desired_goal', 'action', 'next_state', 'achieved_goal', 'reward', 'done'))
env = gym.make("FetchReach-v1")
env.seed(0)
obs = env.reset()
env_params = {'obs_dims': obs['observation'].shape[0],
              'goal_dims': obs['desired_goal'].shape[0],
              'action_dims': env.action_space.shape[0],
              'action_max': env.action_space.high,
              'init_input_means': np.array(([1.309, 0.797, 0.522, 3.96e-6, 1.43e-06, -6.68e-04, 9.73e-04, -2.51e-04,
                                             5.53e-04, 5.69e-4, 1.342, 0.749, 0.535])),
              'init_input_var': np.array(([0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98,
                                           0.98, 0.98, 0.98]))
              }
agent = DDPGAgent(env_params, T, path=path, torch_seed=300, random_seed=300)
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
                env.render(mode="human")
                action = agent.act(obs['observation'], obs['desired_goal'], test=True)
                new_obs, reward, done, info = env.step(action)
                ep_return += reward
                agent.remember(new_episode,
                               obs['observation'], obs['desired_goal'], action,
                               new_obs['observation'], new_obs['achieved_goal'], reward, 1-int(done))
                new_episode = False
                obs = new_obs
            agent.normalizer.update_mean()
            ep += 1
            cycle_return += ep_return
        success_rate = 100*(cycle_return+EPISODE*50)/cycle_timesteps
        success_rates.append(success_rate)
        cycle_returns.append(cycle_return)
        print("Epoch %i" % epo, "cycle %i" % cyc,
              "return %0.1f" % cycle_return, "success rate %0.2f" % success_rate + "%")
        agent.apply_hindsight()
        agent.learn()

    if (epo % 50 == 0) and (epo != 0):
        agent.save_networks(epo)

smoothed_plot("success_rates.png", success_rates, x_label="Cycle")
smoothed_plot("cycle_returns.png", cycle_returns, x_label="Cycle")