import os
import plot
import pybullet_envs
from agent import TD3
algo_params = {
    'prioritised': False,
    'memory_capacity': int(1e6),
    'learning_rate': 0.001,
    'batch_size': 128,
    'optimization_steps': 1,
    'tau': 0.005,
    'discount_factor': 0.98,
    'discard_time_limit': True,

    'update_interval': 50,
    'actor_update_interval': 2,

    'training_episodes': 151,
    'testing_gap': 10,
    'testing_episodes': 10,
    'saving_gap': 50,
}
seeds = [11, 22, 33, 44, 55, 66]
seed_returns = []
path = os.path.dirname(os.path.realpath(__file__))

for seed in seeds:

    env = pybullet_envs.make("InvertedPendulumSwingupBulletEnv-v0")

    seed_path = path + '/seed'+str(seed)

    agent = TD3(algo_params=algo_params, env=env, path=seed_path, seed=seed)
    agent.run(test=False)
    seed_returns.append(agent.statistic_dict['episode_test_return'])
    del env, agent

return_statistic = plot.get_mean_and_deviation(seed_returns, save_data=True,
                                               file_name=os.path.join(path, 'return_statistic.json'))
plot.smoothed_plot_mean_deviation(path + '/returns.png', return_statistic, x_label='Episode', y_label='Average returns')
