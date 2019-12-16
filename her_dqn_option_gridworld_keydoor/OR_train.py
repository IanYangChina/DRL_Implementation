import os
from her_dqn_option_gridworld_keydoor.trainer import Trainer
from envs.gridworld_one_room import OneRoom
setup = {
    'locked_room_height': 20,
    'locked_room_width': 5,
    'locked_room_num': 3,
    'hall_height': 20,
}
demonstrations = [[0, 3, 6],
                  [1, 4, 7],
                  [2, 5, 8]]
seeds = [30, 66, 12, 25, 50]
path = os.path.dirname(os.path.realpath(__file__))

folder = "/Scale_20_5_3_20_3e4"
current_path = path + folder
if not os.path.isdir(current_path):
    os.mkdir(current_path)
for seed in seeds:
    current_path += "/seed" + str(seed)
    if not os.path.isdir(current_path):
        os.mkdir(current_path)
    env = OneRoom(setup, seed=seed)
    trainer = Trainer(env, current_path, seed=seed, training_epoch=301, demonstrations=demonstrations,
                      act_eps_decay=30000)
    print("Seed {}, training starts......".format(seed))
    trainer.run(act_optimization_steps=3, opt_optimization_steps=3)

folder = "/Scale_20_5_3_20_gsrb"
current_path = path + folder
if not os.path.isdir(current_path):
    os.mkdir(current_path)
for seed in seeds:
    current_path += "/seed" + str(seed)
    if not os.path.isdir(current_path):
        os.mkdir(current_path)
    env = OneRoom(setup, seed=seed)
    trainer = Trainer(env, current_path, seed=seed, training_epoch=301, demonstrations=demonstrations,
                      act_exploration='GSRB')
    print("Seed {}, training starts......".format(seed))
    trainer.run(act_optimization_steps=3, opt_optimization_steps=3)

folder = "/Scale_20_5_3_20_gsrb_demon"
current_path = path + folder
if not os.path.isdir(current_path):
    os.mkdir(current_path)
for seed in seeds:
    current_path += "/seed" + str(seed)
    if not os.path.isdir(current_path):
        os.mkdir(current_path)
    env = OneRoom(setup, seed=seed)
    trainer = Trainer(env, current_path, seed=seed, training_epoch=301, demonstrations=demonstrations,
                      act_exploration='GSRB',
                      use_demonstrator_in_training=True)
    print("Seed {}, training starts......".format(seed))
    trainer.run(act_optimization_steps=3, opt_optimization_steps=3)