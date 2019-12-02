import os
from envs.GridWorld_OneRoomType import OneRoomType
from her_dqn_gridworld_keydoor.train import Trainer
path = os.path.dirname(os.path.realpath(__file__))

setup = {
    'locked_room_height': 10,
    'locked_room_width': 10,
    'locked_room_num': 1,
    'hall_height': 10,
}
env = OneRoomType(setup, seed=2222)
trainer = Trainer(env, path, training_epoch=201, torch_seed=30, random_seed=30)
trainer.print_training_info()
trainer.run()