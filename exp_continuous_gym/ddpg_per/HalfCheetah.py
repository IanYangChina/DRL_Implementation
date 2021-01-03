import os
import plot
import pybullet_envs
from agent.ddpg import DDPG
algo_params = {
    'prioritised': True,
    'memory_capacity': int(1e6),
    'learning_rate': 0.001,
    'update_interval': 1,
    'batch_size': 128,
    'optimization_steps': 1,
    'tau': 0.005,
    'discount_factor': 0.98,
    'discard_time_limit': True,

    'random_action_chance': 0.2,
    'noise_deviation': 0.05,

    'training_episodes': 3,
    'testing_episodes': 30,
    'saving_gap': 1,
}
seeds = [11, 22, 33, 44, 55, 66]
seed_returns = []
path = os.path.dirname(os.path.realpath(__file__))

for seed in seeds:

    env = pybullet_envs.make("HalfCheetahBulletEnv-v0")

    seed_path = path + '/seed'+str(seed)

    agent = DDPG(algo_params=algo_params, env=env, path=seed_path, seed=seed)
    agent.run(test=False)
    seed_returns.append(agent.statistic_dict['ep_return'])

return_statistic = plot.get_mean_and_deviation(seed_returns, save_data=True,
                                               file_name=os.path.join(path, 'return_statistic.json'))
plot.smoothed_plot_mean_deviation(path + '/returns.png', return_statistic, x_label='Cycle', y_label='Average returns')
