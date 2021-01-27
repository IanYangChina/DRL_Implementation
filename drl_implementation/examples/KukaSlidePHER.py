# this example runs a goal-condition soft actor critic with prioritised+hindsight experience replay
#       on the Slide task via the pybullet-multigoal-gym package

import os
import gym
import pybullet_multigoal_gym as pmg
from drl_implementation import GoalConditionedSAC, GoalConditionedDDPG
# you can replace the agent instantiation by one of the two above, with the proper params
from drl_implementation.agent.utils import plot
# ddpg_params = {
#     'hindsight': True,
#     'her_sampling_strategy': 'future',
#     'prioritised': True,
#     'memory_capacity': int(1e6),
#     'actor_learning_rate': 0.001,
#     'critic_learning_rate': 0.001,
#     'Q_weight_decay': 0.0,
#     'update_interval': 1,
#     'batch_size': 128,
#     'optimization_steps': 40,
#     'tau': 0.05,
#     'discount_factor': 0.98,
#     'clip_value': 50,
#     'discard_time_limit': True,
#     'terminate_on_achieve': False,
#     'observation_normalization': True,
#
#     'random_action_chance': 0.2,
#     'noise_deviation': 0.05,
#
#     'training_epochs': 101,
#     'training_cycles': 50,
#     'training_episodes': 16,
#     'testing_gap': 1,
#     'testing_episodes': 30,
#     'saving_gap': 25,
# }
sac_params = {
    'hindsight': True,
    'her_sampling_strategy': 'future',
    'prioritised': True,
    'memory_capacity': int(1e6),
    'actor_learning_rate': 0.001,
    'critic_learning_rate': 0.001,
    'update_interval': 1,
    'batch_size': 128,
    'optimization_steps': 40,
    'tau': 0.005,
    'clip_value': 50,
    'discount_factor': 0.98,
    'discard_time_limit': True,
    'terminate_on_achieve': False,
    'observation_normalization': True,

    'alpha': 0.5,
    'actor_update_interval': 1,
    'critic_target_update_interval': 1,

    'training_epochs': 101,
    'training_cycles': 50,
    'training_episodes': 16,
    'testing_gap': 1,
    'testing_episodes': 30,
    'saving_gap': 25,
}
seeds = [11, 22, 33, 44]
seed_returns = []
seed_success_rates = []
path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, 'Slide_PHER')

for seed in seeds:

    env = pmg.make('KukaParallelGripSlideSparseEnv-v0')
    # env = gym.make('FetchReach-v1')

    seed_path = path + '/seed'+str(seed)

    agent = GoalConditionedSAC(algo_params=sac_params, env=env, path=seed_path, seed=seed)
    agent.run(test=False)
    seed_returns.append(agent.statistic_dict['epoch_test_return'])
    seed_success_rates.append(agent.statistic_dict['epoch_test_success_rate'])
    del env, agent

return_statistic = plot.get_mean_and_deviation(seed_returns, save_data=True,
                                               file_name=os.path.join(path, 'return_statistic.json'))
plot.smoothed_plot_mean_deviation(path + '/returns.png', return_statistic, x_label='Epoch', y_label='Average returns')


success_rate_statistic = plot.get_mean_and_deviation(seed_success_rates, save_data=True,
                                                     file_name=os.path.join(path, 'success_rate_statistic.json'))
plot.smoothed_plot_mean_deviation(path + '/success_rates.png', success_rate_statistic,
                                  x_label='Epoch', y_label='Success rates')
