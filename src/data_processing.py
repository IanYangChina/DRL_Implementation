import os
import plot
import json

path = os.getcwd()
path = path[:-4]

return_ddpg_her = json.load(open(os.path.join(path, 'exp_multi_goal', 'ddpg_pybullet_gym', 'Reach_HER', 'return_statistic.json')))
return_ddpg_pher = json.load(open(os.path.join(path, 'exp_multi_goal', 'ddpg_pybullet_gym', 'Reach_PHER', 'return_statistic.json')))
return_sac_her = json.load(open(os.path.join(path, 'exp_multi_goal', 'sac_pybullet_gym', 'Reach_HER', 'return_statistic.json')))
return_sac_pher = json.load(open(os.path.join(path, 'exp_multi_goal', 'sac_pybullet_gym', 'Reach_PHER', 'return_statistic.json')))

plot.smoothed_plot_mean_deviation(os.path.join(path, 'src', 'returns_pybullet_kuka_reach.png'),
                                  [return_ddpg_her, return_ddpg_pher, return_sac_her, return_sac_pher],
                                  legend=['DDPG HER', 'DDPG PHER', 'SAC HER', 'SAC PHER'],
                                  legend_loc='lower right',
                                  title='Pybullet Kuka Reach (6 seeds)',
                                  window=20,
                                  x_label='Cycle', y_label='Average returns')

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

return_ddpg = json.load(open(os.path.join(path, 'exp_continuous_gym', 'ddpg', 'return_statistic.json')))
return_ddpg_per = json.load(open(os.path.join(path, 'exp_continuous_gym', 'ddpg_per', 'return_statistic.json')))
return_sac = json.load(open(os.path.join(path, 'exp_continuous_gym', 'sac', 'return_statistic.json')))
return_sac_per = json.load(open(os.path.join(path, 'exp_continuous_gym', 'sac_per', 'return_statistic.json')))
return_sac_time_limit = json.load(open(os.path.join(path, 'exp_continuous_gym', 'sac_discard_time_limite', 'return_statistic.json')))

plot.smoothed_plot_mean_deviation(os.path.join(path, 'src', 'returns_half_cheetah.png'),
                                  [return_ddpg, return_ddpg_per, return_sac, return_sac_per, return_sac_time_limit],
                                  legend=['DDPG', 'DDPG PER', 'SAC', 'SAC PER', 'SAC -NoTimeLimit'],
                                  legend_loc='lower right',
                                  title='Pybullet Half Cheetah (6 seeds)',
                                  window=10,
                                  x_label='Episode', y_label='Average returns')