import os
import plot
import gym
from agent import DQN
algo_params = {
    'prioritised': True,
    'memory_capacity': int(1e6),
    'critic_learning_rate': 0.00025,
    'update_interval': 16,
    'batch_size': 32,
    'optimization_steps': 1,
    'discount_factor': 0.99,
    'discard_time_limit': False,
    'warmup_step': int(1e4),

    'frame_skip': 4,
    'image_size': 84,
    'epsilon_decay_fraction': 0.02,
    'reward_clip': 1,
    'Q_target_update_interval': int(4e4),
    'Q_weight_decay': 0.95,
    'RMSprop_epsilon':  0.01 / 32**2,
    'gradient_bound': 1. / 32,

    'training_epoch': 20,
    'training_frame_per_epoch': int(1e5),
    'printing_gap': int(1e4),
    'testing_gap': 1,
    'testing_frame_per_epoch': int(1e4),
    'saving_gap': 10,

    'cuda_device_id': 0,
}
seeds = [11, 22, 33, 44, 55, 66]
seed_returns = []
path = os.path.dirname(os.path.realpath(__file__))
for seed in seeds:

    env = gym.make("PongNoFrameskip-v4")
    # env.render()

    seed_path = path + '/seed'+str(seed)

    agent = DQN(algo_params=algo_params, env=env, path=seed_path, seed=seed)
    agent.run(test=False)
    seed_returns.append(agent.statistic_dict['episode_return'])
    del env, agent

return_statistic = plot.get_mean_and_deviation(seed_returns, save_data=True,
                                               file_name=os.path.join(path, 'return_statistic.json'))
plot.smoothed_plot_mean_deviation(path + '/returns.png', return_statistic, x_label='Episode', y_label='Average returns')
