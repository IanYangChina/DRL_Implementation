import numpy as np
import random as r
from envs.gridworld import GridWorldEnv


class TwoRoomEasy(GridWorldEnv):
    """
    This world has two types of rooms: middle and final.
    Keys are all randomly placed in the hall.
    Number of middle rooms and final rooms are equal.
    """
    def __init__(self, env_setup, random_key=True, seed=2222):
        setup = env_setup.copy()
        setup['init_height'] = setup['main_room_height']
        setup['init_width'] = setup['locked_room_num'] * (setup['locked_room_width'] + 1) - 1
        GridWorldEnv.__init__(self, setup, seed)
        self.env_type = "TRE"
        self.random_key = random_key

    def _create_world(self, setup):
        locked_room_width = setup['locked_room_width']
        locked_room_height = setup['locked_room_height']
        locked_room_num = setup['locked_room_num']
        init_room_height = setup['main_room_height']

        # create the main room
        init_room_width = locked_room_width*locked_room_num + locked_room_num-1
        init_room = np.ones((init_room_height, init_room_width), dtype=np.int)
        # create walls and boundaries
        horizontal_wall = np.zeros((1, init_room_width), dtype=np.int)
        horizontal_boundary = np.zeros((1, init_room_width+2), dtype=np.int)
        vertical_wall = np.zeros((locked_room_height, 1), dtype=np.int)
        vertical_boundary = np.zeros((init_room_height+locked_room_height*2+2, 1), dtype=np.int)
        # create locked rooms
        locked_room = np.ones((locked_room_height, locked_room_width), dtype=np.int)
        locked_rooms = np.concatenate((locked_room, vertical_wall), axis=1)
        for mr in range(locked_room_num-1):
            locked_rooms = np.concatenate((locked_rooms, locked_room, vertical_wall), axis=1)
        locked_rooms = np.delete(locked_rooms, -1, 1)

        # concatenate them together, transform into a list of strings
        world_np = np.concatenate((locked_rooms, horizontal_wall, locked_rooms, horizontal_wall, init_room), axis=0)
        world_np = np.concatenate((vertical_boundary, world_np, vertical_boundary), axis=1)
        world_np = np.concatenate((horizontal_boundary, world_np, horizontal_boundary), axis=0)
        world_str = list(world_np.tolist())
        # create a dictionary for goals (key & doors)
        key_door_dict = dict.fromkeys(["mk"+str(i) for i in range(locked_room_num)] +
                                      ["fk"+str(i) for i in range(locked_room_num)] +
                                      ["md" + str(i) for i in range(locked_room_num)] +
                                      ["fd"+str(i) for i in range(locked_room_num)] +
                                      ["fg"+str(i) for i in range(locked_room_num)])

        # calculate the door coordinate for the first middle room
        middle_door_x = locked_room_width // 2
        kls = []
        for _ in range(locked_room_num):
            # using 'mdi' to represent a door in the main room
            key_door_dict['md'+str(_)] = [locked_room_height*2+2, middle_door_x+1*(_+1) + locked_room_width*_]
            world_str[key_door_dict['md'+str(_)][0]][key_door_dict['md'+str(_)][1]] = 'md'+str(_)
            # randomly place the key for each middle room within the main room
            done = False
            while not done:
                mkl = (r.randint(0, init_room_height-1), r.randint(1, init_room_width-1))
                if mkl not in kls:
                    kls.append(mkl)
                    key_door_dict['mk'+str(_)] = [mkl[0]+locked_room_height*2+3, mkl[1]+1]
                    world_str[key_door_dict['mk'+str(_)][0]][key_door_dict['mk'+str(_)][1]] = 'mk'+str(_)
                    done = True
        # calculate the door coordinate for each final room within a middle room
        final_door_x = locked_room_width // 2
        for _ in range(locked_room_num):
            # using 'fdi' to represent a door of a final room
            key_door_dict['fd'+str(_)] = [locked_room_height+1,
                                          final_door_x+1*(_+1) + locked_room_width*_]
            world_str[key_door_dict['fd'+str(_)][0]][key_door_dict['fd'+str(_)][1]] = 'fd'+str(_)

            # randomly choose a cell in a final room to be a final goal
            fg_xy = (r.randint(0, locked_room_height-1), r.randint(0, locked_room_width-1))
            key_door_dict['fg'+str(_)] = [fg_xy[0]+1,
                                          fg_xy[1]+1*(_+1) + locked_room_width*_]
            world_str[key_door_dict['fg'+str(_)][0]][key_door_dict['fg'+str(_)][1]] = 'fg'+str(_)

            # randomly place the key of each final room within the main room or a middle room
            done = False
            while not done:
                fkl = (r.randint(0, init_room_height-1), r.randint(1, init_room_width-1))
                if fkl not in kls:
                    kls.append(fkl)
                    key_door_dict['fk'+str(_)] = [fkl[0]+locked_room_height*2+3, fkl[1]+1]
                    world_str[key_door_dict['fk'+str(_)][0]][key_door_dict['fk'+str(_)][1]] = 'fk'+str(_)
                    done = True

        world_dict = dict.fromkeys(["row"+str(len(world_np)-i-1) for i in range(len(world_np))])
        for i in range(len(world_str)):
            world_dict["row"+str(i)] = world_str[-i-1]
        for n in key_door_dict.items():
            n[1][0] = len(world_dict) - n[1][0] - 1

        return world_dict, key_door_dict

    def _reset_keys(self):
        locked_room_width = self.env_setup['locked_room_width']
        locked_room_height = self.env_setup['locked_room_height']
        locked_room_num = self.env_setup['locked_room_num']
        init_room_height = self.env_setup['main_room_height']
        init_room_width = locked_room_width*locked_room_num + locked_room_num-1
        kls = []
        for _ in range(locked_room_num):
            old_mk_x = self.key_door_dict['mk'+str(_)][1]
            old_mk_y = self.key_door_dict['mk'+str(_)][0]
            self.world["row"+str(old_mk_y)][old_mk_x] = 1
            old_fk_x = self.key_door_dict['fk'+str(_)][1]
            old_fk_y = self.key_door_dict['fk'+str(_)][0]
            self.world["row"+str(old_fk_y)][old_fk_x] = 1
            old_fg_x = self.key_door_dict['fg'+str(_)][1]
            old_fg_y = self.key_door_dict['fg'+str(_)][0]
            self.world["row"+str(old_fg_y)][old_fg_x] = 1

            done = False
            while not done:
                mkl = (r.randint(1, init_room_height),
                       r.randint(1, init_room_width))
                if mkl not in kls:
                    kls.append(mkl)
                    self.key_door_dict['mk'+str(_)] = [mkl[0],
                                                       mkl[1]]
                    self.world['row'+str(self.key_door_dict['mk'+str(_)][0])][self.key_door_dict['mk'+str(_)][1]] = 'mk'+str(_)
                    done = True

            done = False
            while not done:
                fkl = (r.randint(1, init_room_height),
                       r.randint(1, init_room_width))
                if fkl not in kls:
                    kls.append(fkl)
                    self.key_door_dict['fk'+str(_)] = [fkl[0],
                                                       fkl[1]]
                    self.world['row' + str(self.key_door_dict['fk'+str(_)][0])][self.key_door_dict['fk' + str(_)][1]] = 'fk' + str(_)

            # randomly choose a cell in a final room to be a final goal
            fg_xy = (r.randint(init_room_height+locked_room_height+3, init_room_height+2*locked_room_height+2),
                     r.randint(1, locked_room_width))
            self.key_door_dict['fg'+str(_)] = [fg_xy[0],
                                               fg_xy[1]+(locked_room_width+1)*_]
            self.world['row' + str(self.key_door_dict['fg'+str(_)][0])][self.key_door_dict['fg'+str(_)][1]] = 'fg' + str(_)