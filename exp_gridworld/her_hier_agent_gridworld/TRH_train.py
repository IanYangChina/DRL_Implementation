import os
from exp_gridworld.her_hier_agent_gridworld.trainer import Trainer
from envs.gridworld_two_rooms_hard import TwoRoomHard
setup = {
    'locked_room_height': 5,
    'locked_room_width': 5,
    'locked_room_num': 1,
    'main_room_height': 10,
}
demonstrations = [[0, 2, 1, 3, 4]]
seeds = [30, 66, 12, 25, 50]
path = os.path.dirname(os.path.realpath(__file__))

folder = "/Scale_20_7_7_1_1_3e4"
current_path = path + folder
if not os.path.isdir(current_path):
    os.mkdir(current_path)
for seed in seeds:
    current_path += "/seed"+str(seed)
    if not os.path.isdir(current_path):
        os.mkdir(current_path)
    env = TwoRoomHard(setup, seed=seed)
    trainer = Trainer(env, current_path, seed=seed, training_epoch=301, demonstrations=demonstrations,
                      act_eps_decay=30000)
    print("Seed {}, training starts......".format(seed))
    trainer.run(act_optimization_steps=3, opt_optimization_steps=3)

folder = "/Scale_20_7_7_1_1_gsrb"
current_path = path + folder
if not os.path.isdir(current_path):
    os.mkdir(current_path)
for seed in seeds:
    current_path += "/seed"+str(seed)
    if not os.path.isdir(current_path):
        os.mkdir(current_path)
    env = TwoRoomHard(setup, seed=seed)
    trainer = Trainer(env, current_path, seed=seed, training_epoch=301, demonstrations=demonstrations,
                      act_exploration='GSRB')
    print("Seed {}, training starts......".format(seed))
    trainer.run(act_optimization_steps=3, opt_optimization_steps=3)

folder = "/Scale_20_7_7_1_1_gsrb_demon"
current_path = path + folder
if not os.path.isdir(current_path):
    os.mkdir(current_path)
for seed in seeds:
    current_path += "/seed"+str(seed)
    if not os.path.isdir(current_path):
        os.mkdir(current_path)
    env = TwoRoomHard(setup, seed=seed)
    trainer = Trainer(env, current_path, seed=seed, training_epoch=301, demonstrations=demonstrations,
                      act_exploration='GSRB',
                      use_demonstrator_in_training=True)
    print("Seed {}, training starts......".format(seed))
    trainer.run(act_optimization_steps=3, opt_optimization_steps=3)
