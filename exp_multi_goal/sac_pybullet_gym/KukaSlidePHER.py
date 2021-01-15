import os
import plot
import pybullet_multigoal_gym as pmg
from agent import GoalConditionedSAC
algo_params = {
    'hindsight': True,
    'prioritised': True,
    'memory_capacity': int(1e6),
    'actor_learning_rate': 0.0003,
    'critic_learning_rate': 0.0003,
    'update_interval': 1,
    'batch_size': 128,
    'optimization_steps': 40,
    'tau': 0.005,
    'discount_factor': 0.98,
    'discard_time_limit': True,
    'observation_normalization': True,

    'alpha': 0.5,
    'actor_update_interval': 1,
    'critic_target_update_interval': 1,

    'training_epochs': 51,
    'training_cycles': 50,
    'training_episodes': 16,
    'testing_gap': 1,
    'testing_episodes': 30,
    'saving_gap': 25,
}
seeds = [11, 22, 33, 44, 55, 66]
seed_returns = []
seed_success_rates = []
path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, 'Slide_PHER')

for seed in seeds:

    env = pmg.make("KukaParallelGripSlideSparseEnv-v0")

    seed_path = path + '/seed'+str(seed)

    agent = GoalConditionedSAC(algo_params=algo_params, env=env, path=seed_path, seed=seed)
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
