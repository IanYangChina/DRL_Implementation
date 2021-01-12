import os
import plot
import pybullet_envs
from agent import PPO
algo_params = {
    'memory_capacity': int(1e6),
    'actor_learning_rate': 0.0003,
    'critic_learning_rate': 0.0003,
    'optimization_steps': 1,
    'discount_factor': 0.99,
    'observation_normalization': False,

    'batch_size': 512,
    'mini_batch_size': 64,
    'clip_epsilon': 0.20,
    'value_loss_weight': 0.5,
    'GAE_lambda': 0.95,
    'entropy_loss_weight': 0.0,

    'training_episodes': 101,
    'testing_gap': 10,
    'testing_episodes': 10,
    'saving_gap': 50,
}
seeds = [11]
seed_returns = []
path = os.path.dirname(os.path.realpath(__file__))

for seed in seeds:

    env = pybullet_envs.make("InvertedPendulumSwingupBulletEnv-v0")
    # env.render()

    seed_path = path + '/seed'+str(seed)

    agent = PPO(algo_params=algo_params, env=env, path=seed_path, seed=seed)
    agent.run(test=False)
    seed_returns.append(agent.statistic_dict['episode_return'])
    del env, agent

return_statistic = plot.get_mean_and_deviation(seed_returns, save_data=True,
                                               file_name=os.path.join(path, 'return_statistic.json'))
plot.smoothed_plot_mean_deviation(path + '/returns.png', return_statistic, x_label='Episode', y_label='Average returns')
