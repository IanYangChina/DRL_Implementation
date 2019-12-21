import os
from opion_critic_agent_gridworld.trainer import Trainer
from envs.gridworld_one_room import OneRoom
setup = {
    'locked_room_height': 10,
    'locked_room_width': 5,
    'locked_room_num': 1,
    'main_room_height': 10,
}
demonstrations = [[0, 3, 6],
                  [1, 4, 7],
                  [2, 5, 8]]
seeds = [30, 66, 12, 25, 50]
path = os.path.dirname(os.path.realpath(__file__))

folder = "/Scale_10_5_1_10"
current_path = path + folder
if not os.path.isdir(current_path):
    os.mkdir(current_path)
for seed in seeds:
    current_path += "/seed" + str(seed)
    if not os.path.isdir(current_path):
        os.mkdir(current_path)
    env = OneRoom(setup, seed=seed)
    trainer = Trainer(env, current_path, seed=seed, training_epoch=201)
    print("{}, seed {}, training starts......".format(folder, seed))
    trainer.run(opt_optimization_steps=3)