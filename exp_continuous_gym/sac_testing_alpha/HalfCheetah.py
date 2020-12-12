import os
import plot
from agent.sac_continuous import SACAgent
from exp_continuous_gym.trainer import Trainer
seeds = [11]
seed_returns = []
path = os.path.dirname(os.path.realpath(__file__))

for seed in seeds:

    seed_path = path + '/seed'+str(seed)
    trainer = Trainer(env="HalfCheetahBulletEnv-v0",
                      seed=seed,
                      render=False,
                      path=seed_path,
                      agent=SACAgent,
                      prioritised=False)

    seed_return = trainer.run(test=False, n_episodes=60, load_network_ep=150)
    plot.smoothed_plot(path + 'alpha.png', trainer.agent.alpha_record, x_label='update step', y_label='alpha', window='5')
    plot.smoothed_plot(path + 'policy_entropy.png', trainer.agent.policy_entropy_record, x_label='update step', y_label='policy entropy', window='5')
#     seed_returns.append(seed_return)
#
# return_statistic = plot.get_mean_and_deviation(seed_returns, save_data=True,
#                                                file_name=os.path.join(path, 'return_statistic.json'))
# plot.smoothed_plot_mean_deviation(path + '/returns.png', return_statistic, x_label='Cycle', y_label='Average returns')