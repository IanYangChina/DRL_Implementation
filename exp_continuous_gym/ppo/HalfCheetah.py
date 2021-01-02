import os
import plot
from agent.ppo_continuous import PPOAgent
from exp_continuous_gym.trainer import Trainer
seeds = [11, 22, 33, 44, 55, 66]
seed_returns = []
path = os.path.dirname(os.path.realpath(__file__))

for seed in seeds:

    seed_path = path + '/seed'+str(seed)
    trainer = Trainer(env="HalfCheetahBulletEnv-v0",
                      seed=seed,
                      render=False,
                      path=seed_path,
                      agent=PPOAgent,
                      update_interval=128)

    seed_return = trainer.run(test=False, load_network_ep=150)
    seed_returns.append(seed_return)

return_statistic = plot.get_mean_and_deviation(seed_returns, save_data=True,
                                               file_name=os.path.join(path, 'return_statistic.json'))
plot.smoothed_plot_mean_deviation(path + '/returns.png', return_statistic, x_label='Cycle', y_label='Average returns')