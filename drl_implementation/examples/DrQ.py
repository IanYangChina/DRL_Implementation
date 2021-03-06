# this example runs a DrQ agent on the walker2D task from pybullet-gym

import os
import pybullet_envs
from drl_implementation import SACDrQ

drq_params = {
    'image_crop_size': 140,
    'image_resize_size': 84,
    'frame_stack': 3,
    'prioritised': True,
    'memory_capacity': int(1e5),
    'actor_learning_rate': 0.001,
    'critic_learning_rate': 0.001,
    'update_interval': 1,
    'batch_size': 512,
    'optimization_steps': 1,
    'tau': 0.01,
    'discount_factor': 0.99,
    'discard_time_limit': True,

    'alpha': 0.1,
    'actor_update_interval': 2,
    'critic_target_update_interval': 2,
    'warmup_step': 1000,
    'q_regularisation_k': 2,

    'max_env_step': 200000,
    'testing_gap': 10000,
    'testing_episodes': 10,
    'saving_gap': 100000,
}

seeds = [11]
seed_returns = []
path = os.path.dirname(os.path.realpath(__file__))
for seed in seeds:

    env = pybullet_envs.make("Walker2DBulletEnv-v0")
    # call render() before training to visualize (pybullet-gym-specific)
    # env.render()
    seed_path = path + '/seed'+str(seed)

    agent = SACDrQ(algo_params=drq_params, env=env, path=seed_path, seed=seed)
    agent.run(test=False)
    # the sleep argument pause the rendering for a while at every env step, useful for slowing down visualization
    # agent.run(test=True, load_network_ep=50, sleep=0.05)
    seed_returns.append(agent.statistic_dict['episode_return'])
    del env, agent
