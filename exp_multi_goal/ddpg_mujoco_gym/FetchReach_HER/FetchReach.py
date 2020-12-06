import os
from agent.ddpg_her_continuous import HindsightDDPGAgent
from exp_multi_goal.trainer import Trainer
seeds = [11, 22, 33, 44, 55, 66]

for seed in seeds:

    path = os.path.dirname(os.path.realpath(__file__))
    path += '/seed'+str(seed)
    if not os.path.isdir(path):
        os.mkdir(path)

    trainer = Trainer(env="FetchReach-v1",
                      render=False,
                      agent=HindsightDDPGAgent,
                      hindsight=True,
                      prioritised=False,
                      seed=seed,
                      path=path)

    trainer.run(test=False, load_network_ep=150)
