from envs.GridWorld_TwoRoomTypes import TwoRoomType
from envs.GridWorld_OneRoomType import OneRoomType


keydoor_setup = {'locked_room_height': 8,
                 'locked_room_width': 5,
                 'locked_room_num': 3,
                 'hall_height': 8}
print("KeyDoor")
env = OneRoomType(keydoor_setup, seed=20)
for key in env.world:
    print(key, env.world[key])
for key in env.key_door_dict:
    print(key, env.key_door_dict[key])


hardkeydoor_setup = {'middle_room_size': 5,
                     'middle_room_num': 3,
                     'final_room_num': 3,
                     'main_room_height': 20}
print("\nHardKeyDoor")
env = TwoRoomType(hardkeydoor_setup, seed=20)
for key in env.world:
    print(key, env.world[key])
for key in env.key_door_dict:
    print(key, env.key_door_dict[key])
