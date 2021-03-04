# this example runs a ddpg agent on a inverted pendulum swingup task from pybullet gym

import os
import pybullet_envs
from drl_implementation import DDPG, SAC, TD3
# you can replace the agent instantiation by one of the three above, with the proper params

# td3_params = {
#     'prioritised': True,
#     'memory_capacity': int(1e6),
#     'actor_learning_rate': 0.0003,
#     'critic_learning_rate': 0.0003,
#     'batch_size': 256,
#     'optimization_steps': 1,
#     'tau': 0.005,
#     'discount_factor': 0.99,
#     'discard_time_limit': True,
#     'warmup_step': 2500,
#     'target_noise': 0.2,
#     'noise_clip': 0.5,
#     'update_interval': 1,
#     'actor_update_interval': 2,
#     'observation_normalization': False,
#
#     'training_episodes': 101,
#     'testing_gap': 10,
#     'testing_episodes': 10,
#     'saving_gap': 50,
# }
# sac_params = {
#     'prioritised': True,
#     'memory_capacity': int(1e6),
#     'actor_learning_rate': 0.0003,
#     'critic_learning_rate': 0.0003,
#     'update_interval': 1,
#     'batch_size': 256,
#     'optimization_steps': 1,
#     'tau': 0.005,
#     'discount_factor': 0.99,
#     'discard_time_limit': True,
#     'observation_normalization': False,
#
#     'alpha': 0.5,
#     'actor_update_interval': 1,
#     'critic_target_update_interval': 1,
#     'warmup_step': 1000,
#
#     'training_episodes': 101,
#     'testing_gap': 10,
#     'testing_episodes': 10,
#     'saving_gap': 50,
# }
ddpg_params = {
    'prioritised': True,
    'memory_capacity': int(1e6),
    'actor_learning_rate': 0.001,
    'critic_learning_rate': 0.001,
    'Q_weight_decay': 0.0,
    'update_interval': 1,
    'batch_size': 100,
    'optimization_steps': 1,
    'tau': 0.005,
    'discount_factor': 0.99,
    'discard_time_limit': True,
    'warmup_step': 2500,
    'observation_normalization': False,

    'training_episodes': 101,
    'testing_gap': 10,
    'testing_episodes': 10,
    'saving_gap': 50,
}
seeds = [11, 22, 33, 44, 55, 66]
seed_returns = []
path = os.path.dirname(os.path.realpath(__file__))
for seed in seeds:

    env = pybullet_envs.make("InvertedPendulumSwingupBulletEnv-v0")
    # call render() before training to visualize (pybullet-gym-specific)
    # env.render()
    seed_path = path + '/seed'+str(seed)

    agent = DDPG(algo_params=ddpg_params, env=env, path=seed_path, seed=seed)
    agent.run(test=False)
    # the sleep argument pause the rendering for a while at every env step, useful for slowing down visualization
    # agent.run(test=True, load_network_ep=50, sleep=0.05)
    seed_returns.append(agent.statistic_dict['episode_return'])
    del env, agent
