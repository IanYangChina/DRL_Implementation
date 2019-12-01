import os
from envs.GridWorld_TwoRoomTypesHard import TwoRoomTypeHard
from her_dqn_option_gridworld_keydoor.train import Trainer
path = os.path.dirname(os.path.realpath(__file__))

env_setup = {'middle_room_size': 5,
             'middle_room_num': 3,
             'final_room_num': 3,
             'main_room_height': 20}
env = TwoRoomTypeHard(env_setup, seed=2222)

trainer = Trainer(env, path, training_epoch=201)
trainer.print_training_info()
trainer.run()
