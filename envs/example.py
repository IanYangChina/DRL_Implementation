from envs.gridworld_one_room import OneRoom
from envs.gridworld_two_rooms_easy import TwoRoomEasy
from envs.gridworld_two_rooms_hard import TwoRoomHard


oneroom_setup = {'locked_room_height': 8,
                 'locked_room_width': 5,
                 'locked_room_num': 3,
                 'hall_height': 8}
env = OneRoom(oneroom_setup, seed=20)
for key in env.world:
    print(key, env.world[key])
for key in env.key_door_dict:
    print(key, env.key_door_dict[key])


tworoomeasy_setup = {'middle_room_size': 5,
                     'middle_room_num': 3,
                     'final_room_num': 3,
                     'main_room_height': 20}
env = TwoRoomEasy(tworoomeasy_setup, seed=20)
for key in env.world:
    print(key, env.world[key])
for key in env.key_door_dict:
    print(key, env.key_door_dict[key])

tworoomhard_setup = {'middle_room_size': 5,
                     'middle_room_num': 3,
                     'final_room_num': 3,
                     'main_room_height': 20}
env = TwoRoomHard(tworoomhard_setup, seed=20)
for key in env.world:
    print(key, env.world[key])
for key in env.key_door_dict:
    print(key, env.key_door_dict[key])