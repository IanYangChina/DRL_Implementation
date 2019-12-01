import numpy as np
import random as r
from envs.GridWorld import GridWorldEnv


class TwoRoomTypeEasy(GridWorldEnv):
    """
    This world has two types of rooms: middle and final.
    Keys are all randomly placed in the hall.
    Number of middle rooms and final rooms could be different, but finals should not be more than middles.
    """
    def __init__(self, env_setup, seed=2222):
        self.env_type = "TRTE"
        setup = env_setup.copy()
        setup['init_height'] = setup['main_room_height']
        setup['init_width'] = setup['middle_room_num']*(setup['middle_room_size']+1)-1
        GridWorldEnv.__init__(self, setup, seed)

    def _create_world(self, setup):
        middle_room_size = setup['middle_room_size']
        middle_room_num = setup['middle_room_num']
        final_room_num = setup['final_room_num']
        init_room_height = setup['main_room_height']
        if middle_room_num < final_room_num:
            raise ValueError("The number of middle rooms should be greater than that of final rooms")
        # create the main room
        init_room_width = middle_room_size*middle_room_num + middle_room_num-1
        init_rooms = np.ones((init_room_height, init_room_width), dtype=np.int)
        # create walls and boundaries
        horizontal_wall = np.zeros((1, init_room_width), dtype=np.int)
        horizontal_boundary = np.zeros((1, init_room_width+2), dtype=np.int)
        vertical_wall = np.zeros((middle_room_size, 1), dtype=np.int)
        vertical_boundary = np.zeros((init_room_height+middle_room_size*2+2, 1), dtype=np.int)
        # create middle rooms
        middle_room = np.ones((middle_room_size, middle_room_size), dtype=np.int)
        middle_rooms = np.concatenate((middle_room, vertical_wall), axis=1)
        for mr in range(middle_room_num-1):
            middle_rooms = np.concatenate((middle_rooms, middle_room, vertical_wall), axis=1)
        middle_rooms = np.delete(middle_rooms, -1, 1)
        # create final rooms
        final_room = np.ones((middle_room_size, middle_room_size), dtype=np.int)
        final_rooms = np.concatenate((final_room, vertical_wall), axis=1)
        if final_room_num > 1:
            for fr in range(final_room_num-1):
                final_rooms = np.concatenate((final_rooms, final_room, vertical_wall), axis=1)
        final_rooms = np.delete(final_rooms, -1, 1)
        # extend the final room matrix with blocked space (np.zeros),
        # if the number of final rooms is less than that of middle rooms
        block_room_num = middle_room_num - final_room_num
        if block_room_num != 0:
            blocked_room = np.zeros((middle_room_size, middle_room_size), dtype=np.int)
            blocked_rooms = np.concatenate((blocked_room, vertical_wall), axis=1)
            if block_room_num > 1:
                for br in range(block_room_num-1):
                    blocked_rooms = np.concatenate((blocked_rooms, blocked_room, vertical_wall), axis=1)
            final_rooms = np.concatenate((blocked_rooms, final_rooms), axis=1)
        # concatenate them together, with an extra str_like list
        world_np = np.concatenate((final_rooms, horizontal_wall, middle_rooms, horizontal_wall, init_rooms), axis=0)
        world_np = np.concatenate((vertical_boundary, world_np, vertical_boundary), axis=1)
        world_np = np.concatenate((horizontal_boundary, world_np, horizontal_boundary), axis=0)
        world_str = list(world_np.tolist())
        # create a dictionary for goals (key & doors)
        key_door_dict = dict.fromkeys(["mk"+str(i) for i in range(middle_room_num)] +
                                      ["fk"+str(i) for i in range(final_room_num)] +
                                      ["md" + str(i) for i in range(middle_room_num)] +
                                      ["fd"+str(i) for i in range(final_room_num)] +
                                      ["fg"+str(i) for i in range(final_room_num)])

        # calculate the door coordinate for the first middle room
        middle_door_x = middle_room_size // 2
        kls = []
        for _ in range(middle_room_num):
            # using 'mdi' to represent a door in the main room
            key_door_dict['md'+str(_)] = [middle_room_size*2+2, middle_door_x+1*(_+1) + middle_room_size*_]
            world_str[key_door_dict['md'+str(_)][0]][key_door_dict['md'+str(_)][1]] = 'md'+str(_)
            # randomly place the key for each middle room within the main room
            done = False
            while not done:
                mkl = (r.randint(0, init_room_height-1), r.randint(1, init_room_width-1))
                if mkl not in kls:
                    kls.append(mkl)
                    key_door_dict['mk'+str(_)] = [mkl[0]+middle_room_size*2+3, mkl[1]+1]
                    world_str[key_door_dict['mk'+str(_)][0]][key_door_dict['mk'+str(_)][1]] = 'mk'+str(_)
                    done = True
        # calculate the door coordinate for each final room within a middle room
        final_door_x = middle_room_size // 2
        for _ in range(final_room_num):
            # using 'fdi' to represent a door of a final room
            key_door_dict['fd'+str(_)] = [middle_room_size+1,
                                          final_door_x+1*(block_room_num+_+1) + middle_room_size*(block_room_num+_)]
            world_str[key_door_dict['fd'+str(_)][0]][key_door_dict['fd'+str(_)][1]] = 'fd'+str(_)

            # randomly choose a cell in a final room to be a final goal
            fg_xy = (r.randint(0, middle_room_size-1), r.randint(0, middle_room_size-1))
            key_door_dict['fg'+str(_)] = [fg_xy[0]+1,
                                          fg_xy[1]+1*(block_room_num+_+1) + middle_room_size*(block_room_num+_)]
            world_str[key_door_dict['fg'+str(_)][0]][key_door_dict['fg'+str(_)][1]] = 'fg'+str(_)

            # randomly place the key of each final room within the main room or a middle room
            done = False
            while not done:
                fkl = (r.randint(0, init_room_height-1), r.randint(1, init_room_width-1))
                if fkl not in kls:
                    kls.append(fkl)
                    key_door_dict['fk'+str(_)] = [fkl[0]+middle_room_size*2+3, fkl[1]+1]
                    world_str[key_door_dict['fk'+str(_)][0]][key_door_dict['fk'+str(_)][1]] = 'fk'+str(_)
                    done = True

        world_dict = dict.fromkeys(["row"+str(len(world_np)-i-1) for i in range(len(world_np))])
        for i in range(len(world_str)):
            world_dict["row"+str(i)] = world_str[-i-1]
        for n in key_door_dict.items():
            n[1][0] = len(world_dict) - n[1][0] - 1

        return world_dict, key_door_dict
