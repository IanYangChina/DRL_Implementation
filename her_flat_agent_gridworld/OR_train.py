import os
from envs.gridworld_one_room import OneRoom
from her_flat_agent_gridworld.trainer import Trainer
path = os.path.dirname(os.path.realpath(__file__))

setup = {
    'locked_room_height': 10,
    'locked_room_width': 10,
    'locked_room_num': 1,
    'main_room_height': 10,
}
env = OneRoom(setup, seed=2222)
trainer = Trainer(env, path, agent_type='dqn', training_epoch=51, seed=2222)
trainer.agent.load_network(epo=50)
# trainer.run()
trainer.test(render=True, given_goal=1)