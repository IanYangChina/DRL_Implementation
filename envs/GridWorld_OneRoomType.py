import numpy as np
import random as r
from envs.GridWorld import GridWorldEnv


class OneRoomType(GridWorldEnv):
    """
    This world has one type of rooms.
    Keys are placed in the hall.
    """
    def __init__(self, env_setup, seed=2222):
        setup = env_setup.copy()
        setup['init_height'] = setup['hall_height']
        setup['init_width'] = setup['locked_room_num']*(setup['locked_room_width']+1)-1
        GridWorldEnv.__init__(self, setup, seed)

    def _create_world(self, setup):
        """
            This function automatically create a world given some information of the world.
            The height of the hall room is manual given.
            The height of the world is determined by the height of the hall room and the size of middle/final rooms.
            The width of the world is determined by the size and number of middle/final rooms.
        """
        locked_room_height = setup['locked_room_height']
        locked_room_width = setup['locked_room_width']
        locked_room_num = setup['locked_room_num']
        hall_height = setup['hall_height']

        # create the hall
        hall_width = locked_room_width*locked_room_num + locked_room_num-1
        hall = np.ones((hall_height, hall_width), dtype=np.int)
        # create walls and boundaries
        horizontal_wall = np.zeros((1, hall_width), dtype=np.int)
        horizontal_boundary = np.zeros((1, hall_width+2), dtype=np.int)
        vertical_wall = np.zeros((locked_room_height, 1), dtype=np.int)
        vertical_boundary = np.zeros((hall_height+locked_room_height+1, 1), dtype=np.int)
        # create locked rooms

        locked_room = np.ones((locked_room_height, locked_room_width), dtype=np.int)
        locked_rooms = np.concatenate((locked_room, vertical_wall), axis=1)
        for mr in range(locked_room_num-1):
            locked_rooms = np.concatenate((locked_rooms, locked_room, vertical_wall), axis=1)
        locked_rooms = np.delete(locked_rooms, -1, 1)

        # concatenate them together, with an extra str_like list
        world_np = np.concatenate((locked_rooms, horizontal_wall, hall), axis=0)
        world_np = np.concatenate((vertical_boundary, world_np, vertical_boundary), axis=1)
        world_np = np.concatenate((horizontal_boundary, world_np, horizontal_boundary), axis=0)
        world_str = list(world_np.tolist())
        # create a dictionary for goals (key & doors)
        key_door_dict = dict.fromkeys(["k"+str(i) for i in range(locked_room_num)] +
                                      ["d"+str(i) for i in range(locked_room_num)] +
                                      ["fg"+str(i) for i in range(locked_room_num)])

        # calculate the door coordinate for the first locked room
        locked_door_x = locked_room_width // 2
        kls = []
        for _ in range(locked_room_num):
            # using 'di' to represent the i-th locked door in the hall
            key_door_dict['d'+str(_)] = [locked_room_height+1, locked_door_x+1*(_+1)+locked_room_width*_]
            world_str[key_door_dict['d'+str(_)][0]][key_door_dict['d'+str(_)][1]] = 'd'+str(_)
            # using 'fgi' to represent the i-th final goal in the i-th locked room
            fg_xy = (r.randint(0, locked_room_height - 1), r.randint(0, locked_room_width - 1))
            key_door_dict['fg'+str(_)] = [fg_xy[0]+1, fg_xy[1] + 1*(_+1) + locked_room_width*_]
            world_str[key_door_dict['fg'+str(_)][0]][key_door_dict['fg'+str(_)][1]] = 'fg'+str(_)
            # randomly place the key for each locked room within the hall
            done = False
            while not done:
                kl = (r.randint(0, hall_height-1), r.randint(1, hall_width-1))
                if kl not in kls:
                    kls.append(kl)
                    key_door_dict['k'+str(_)] = [kl[0]+locked_room_height+2, kl[1]+1]
                    world_str[key_door_dict['k'+str(_)][0]][key_door_dict['k'+str(_)][1]] = 'k' + str(_)
                    done = True

        world_dict = dict.fromkeys(["row"+str(len(world_np)-i-1) for i in range(len(world_np))])
        for i in range(len(world_str)):
            world_dict["row"+str(i)] = world_str[-i-1]
        for n in key_door_dict.items():
            n[1][0] = len(world_dict)-n[1][0]-1

        return world_dict, key_door_dict
