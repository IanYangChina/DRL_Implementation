import os
import plot
import json

path = os.getcwd()
path = path[:-4]

# return_ddpg_her = json.load(open(os.path.join(path, 'exp_multi_goal', 'ddpg_pybullet_gym', 'Reach_HER', 'return_statistic.json')))
# return_ddpg_pher = json.load(open(os.path.join(path, 'exp_multi_goal', 'ddpg_pybullet_gym', 'Reach_PHER', 'return_statistic.json')))
# return_sac_her = json.load(open(os.path.join(path, 'exp_multi_goal', 'sac_pybullet_gym', 'Reach_HER', 'return_statistic.json')))
# return_sac_pher = json.load(open(os.path.join(path, 'exp_multi_goal', 'sac_pybullet_gym', 'Reach_PHER', 'return_statistic.json')))
#
# plot.smoothed_plot_mean_deviation(os.path.join(path, 'src', 'returns_pybullet_kuka_reach.png'),
#                                   [return_ddpg_her, return_ddpg_pher, return_sac_her, return_sac_pher],
#                                   legend=['DDPG HER', 'DDPG PHER', 'SAC HER', 'SAC PHER'],
#                                   legend_loc='lower right',
#                                   title='Pybullet Kuka Reach (6 seeds)',
#                                   window=20,
#                                   x_label='Cycle', y_label='Average returns')

# success_rate_ddpg_her = json.load(open(os.path.join(path, 'exp_multi_goal', 'ddpg_pybullet_gym', 'Reach_HER', 'success_rate_statistic.json')))
# success_rate_ddpg_pher = json.load(open(os.path.join(path, 'exp_multi_goal', 'ddpg_pybullet_gym', 'Reach_PHER', 'success_rate_statistic.json')))
# success_rate_sac_her = json.load(open(os.path.join(path, 'exp_multi_goal', 'sac_pybullet_gym', 'Reach_HER', 'success_rate_statistic.json')))
# success_rate_sac_pher = json.load(open(os.path.join(path, 'exp_multi_goal', 'sac_pybullet_gym', 'Reach_PHER', 'success_rate_statistic.json')))
#
# plot.smoothed_plot_mean_deviation(os.path.join(path, 'src', 'success_rates.png'),
#                                   [success_rate_ddpg_her, success_rate_ddpg_pher, success_rate_sac_her, success_rate_sac_pher],
#                                   legend=['DDPG HER', 'DDPG PHER', 'SAC HER', 'SAC PHER'],
#                                   legend_loc='lower right',
#                                   window=20,
#                                   x_label='Cycle', y_label='Average success_rates')

# return_ddpg = json.load(open(os.path.join(path, 'exp_continuous_gym', 'pendulum', 'ddpg', 'return_statistic.json')))
# return_td3 = json.load(open(os.path.join(path, 'exp_continuous_gym', 'pendulum', 'td3', 'return_statistic.json')))
# return_sac = json.load(open(os.path.join(path, 'exp_continuous_gym', 'pendulum', 'sac', 'return_statistic.json')))
#
# plot.smoothed_plot_mean_deviation(os.path.join(path, 'src', 'returns_pendulum.png'),
#                                   [return_ddpg, return_td3, return_sac],
#                                   legend=['DDPG', 'TD3', 'SAC'],
#                                   legend_loc='lower right',
#                                   title='Pybullet Inverted Pendulum Swingup (6 seeds)',
#                                   window=10,
#                                   x_label='Episode', y_label='Average returns')