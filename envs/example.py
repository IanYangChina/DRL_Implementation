from envs.gridworld_one_room import OneRoom
from envs.gridworld_two_rooms_easy import TwoRoomEasy
from envs.gridworld_two_rooms_hard import TwoRoomHard
oneroom_setup = {'locked_room_height': 8,
                 'locked_room_width': 5,
                 'locked_room_num': 3,
                 'main_room_height': 8}
env0 = OneRoom(oneroom_setup, random_key=False, seed=20)
env0.render()

tworoomeasy_setup = {'locked_room_height': 8,
                     'locked_room_width': 5,
                     'locked_room_num': 3,
                     'main_room_height': 8}
env1 = TwoRoomEasy(tworoomeasy_setup, random_key=False, seed=20)
env1.render()

tworoomhard_setup = {'locked_room_height': 8,
                     'locked_room_width': 5,
                     'locked_room_num': 3,
                     'main_room_height': 8}
env2 = TwoRoomHard(tworoomhard_setup, random_key=False, seed=20)
env2.render()