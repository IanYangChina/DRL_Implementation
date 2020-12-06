import os
from agent.ddpg_continuous import DDPGAgent
from exp_continuous_gym.trainer import Trainer
seeds = [11, 22, 33, 44, 55, 66]

for seed in seeds:

    path = os.path.dirname(os.path.realpath(__file__))
    path += '/seed'+str(seed)
    if not os.path.isdir(path):
        os.mkdir(path)

    trainer = Trainer(env="HalfCheetahBulletEnv-v0",
                      agent=DDPGAgent,
                      prioritised=True,
                      seed=seed,
                      render=False,
                      path=path)

    trainer.run(test=False, load_network_ep=150)
