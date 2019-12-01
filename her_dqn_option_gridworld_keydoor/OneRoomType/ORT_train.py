import os
from envs.GridWorld_OneRoomType import OneRoomType
from her_dqn_option_gridworld_keydoor.train import Trainer

path = os.path.dirname(os.path.realpath(__file__))

setup = {
    'locked_room_height': 15,
    'locked_room_width': 3,
    'locked_room_num': 2,
    'hall_height': 20,
}

env = OneRoomType(setup, seed=2222)
trainer = Trainer(env, path, training_epoch=201)
trainer.print_training_info()
trainer.run()
