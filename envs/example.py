from envs.gridworld_one_room import OneRoom
from envs.gridworld_two_rooms_easy import TwoRoomEasy
from envs.gridworld_two_rooms_hard import TwoRoomHard


oneroom_setup = {'locked_room_height': 8,
                 'locked_room_width': 5,
                 'locked_room_num': 3,
                 'main_room_height': 8}
env = OneRoom(oneroom_setup, random_key=False, seed=20)
for key in env.world:
    print(key, env.world[key])
for key in env.key_door_dict:
    print(key, env.key_door_dict[key])

tworoomeasy_setup = {'locked_room_height': 8,
                     'locked_room_width': 5,
                     'locked_room_num': 3,
                     'main_room_height': 8}
env = TwoRoomEasy(tworoomeasy_setup, random_key=False, seed=20)
for key in env.world:
    print(key, env.world[key])
for key in env.key_door_dict:
    print(key, env.key_door_dict[key])

tworoomhard_setup = {'locked_room_height': 8,
                     'locked_room_width': 5,
                     'locked_room_num': 3,
                     'main_room_height': 8}
env = TwoRoomHard(tworoomhard_setup, random_key=False, seed=20)
for key in env.world:
    print(key, env.world[key])
for key in env.key_door_dict:
    print(key, env.key_door_dict[key])