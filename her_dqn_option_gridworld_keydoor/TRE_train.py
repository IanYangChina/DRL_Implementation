import os
from envs.gridworld_two_rooms_easy import TwoRoomEasy
from her_dqn_option_gridworld_keydoor.trainer import Trainer
path = os.path.dirname(os.path.realpath(__file__))

env_setup = {'middle_room_size': 5,
             'middle_room_num': 3,
             'final_room_num': 3,
             'main_room_height': 20}
folder = '/TwoRoomEasy'
path += folder
if not os.path.isdir(path):
    os.mkdir(path)
env = TwoRoomEasy(env_setup, seed=2222)
trainer = Trainer(env, path, training_epoch=201)
trainer.print_training_info()
trainer.run()
