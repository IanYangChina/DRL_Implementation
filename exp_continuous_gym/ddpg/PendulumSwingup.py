import os
import plot
import pybullet_envs
from agent import DDPG
algo_params = {
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
    # env.render()

    seed_path = path + '/seed'+str(seed)

    agent = DDPG(algo_params=algo_params, env=env, path=seed_path, seed=seed)
    agent.run(test=False)
    seed_returns.append(agent.statistic_dict['episode_return'])
    del env, agent

return_statistic = plot.get_mean_and_deviation(seed_returns, save_data=True,
                                               file_name=os.path.join(path, 'return_statistic.json'))
plot.smoothed_plot_mean_deviation(path + '/returns.png', return_statistic, x_label='Episode', y_label='Average returns')
