import numpy as np
import random as r
from copy import deepcopy as dcp

r.seed(2222)


class Env(object):
    """
    --Brief Introduction:
        An grid-world constituted by three types of rooms: a hall, middle rooms and final rooms.
        When calling function reset(), an agent will be initialized at a random cell within the hall.
        In order to enter middle rooms or final rooms, an agent needs to first collect the corresponding keys.
        Keys for entering middle rooms are randomly placed within the hall.
        Keys for entering final rooms are randomly placed within middle rooms.
        The ultimate goal-locations are cells of the final rooms.

    --MDP Definition:
        States and goals are represented by [x, y] coordinates of the cells.
        Observation is constituted by a state, a goal and an inventory of keys.
            The inventory is a one-hot vector whose size is the number of keys in an environment instance.
        Primitive actions are "up", "down", "left" and "right" (deterministic actions).
            Reward for primitive-agent is 1 when a desired goal given by the option-agent is achieved and 0 otherwise.
        Options are coordinates of goals.
            Reward for option-agent is 1 when an ultimate goal is achieved and 0 otherwise.

    --Challenge:
        1. Exploration efficiency.
            This is due to the inter-connections between goals (keys and doors).
            Specifically, consider the simplest instance with 1 middle and final room, the agent needs to reach 4 goals
                (middle key, middle door, final key, final door) in a specific order, before it has a chance to reach
                the ultimate goal.
            Since the difficulty of entering a room is increasing from middle rooms to final rooms, the exploration of
                rooms are also increasing.
            Thus, achieving an ultimate goal is fairly difficult, and it is necessary to introduce better ideas that
                improve exploration instead of solely counting on random actions.
        2. Sample efficiency.
            This is a common issue bothering RL algorithms in sparse reward environments.
            However, in this environment where goals are inter-related to each other, obtaining experiences valuable
                for learning is even more unusual.
            Note that, Hindsight Experience Replay is helpful for training the low level agent, while it is not so for
                training the high level agent. Since the reward is only obtained when achieving an ultimate goal, not
                an arbitrary goal as the case of low level agent.
    """
    def __init__(self, init_room_height, middle_room_size, middle_room_num, final_room_num):
        self.init_room_height = init_room_height
        self.init_room_width = middle_room_num*middle_room_size + middle_room_num-1
        self.world, self.key_door_dict = self._create_world(init_room_height, middle_room_size,
                                                            middle_room_num, final_room_num)

        self.keys = [key for key in self.key_door_dict if "k" in key]
        self.doors = [key for key in self.key_door_dict if "d" in key]
        self.final_goals = [key for key in self.key_door_dict if "fg" in key]
        self.goal_space = self.keys + self.doors + self.final_goals

        self.option_space = [op for op in range(len(self.goal_space))]
        self.action_space = [0, 1, 2, 3]
        self.actions = ['up', 'down', 'left', 'right']

        self.keys_running = None
        self.world_running = None

        self.input_max = np.array(([self.init_room_width, self.init_room_height+2*middle_room_size+2,
                                    self.init_room_width,
                                    self.init_room_height+2*middle_room_size+2]+[1 for k in range(len(self.keys))]),
                                  dtype=np.float)
        self.input_min = np.array(([1, 1, 1, 1]+[0 for k in range(len(self.keys))]), dtype=np.float)

    def reset(self, all_goal=False, single=False, act_test=False):
        """
        Every time an episode ends, call this function.

        :param all_goal:  If True, all of the goals are possible to be the high level agent's desired goal.
        :param single:    If True, the high level agent only pursue one fixed goal
        :param act_test:  If True, this function only return low level observations (used for test the low level agent)
        :return:          Initial observations
        """
        self.world_running = dcp(self.world)
        self.keys_running = dcp(self.keys)
        x, y = r.randint(1, self.init_room_width), r.randint(1, self.init_room_height)
        if not all_goal:
            final_goal = r.choice(self.final_goals)
        else:
            final_goal = r.choice(self.final_goals + self.doors + self.keys)
        if single:
            fianl_goal = self.doors[0]

        final_goal_loc = np.array(([self.key_door_dict[final_goal][1], self.key_door_dict[final_goal][2]]), dtype=np.float)
        achieved_goal = self.world_running['row'+str(y)][x]
        achieved_goal_loc = np.array(([x, y]), dtype=np.float)
        act_observation = {'state': np.array(([x, y]), dtype=np.float),
                           'inventory': [],
                           'inventory_vector': np.zeros(len(self.keys), dtype=np.float),
                           'desired_goal': final_goal,
                           'desired_goal_loc': final_goal_loc,
                           'achieved_goal': achieved_goal,
                           'achieved_goal_loc': achieved_goal_loc}
        opt_observation = {'state': np.array(([x, y]), dtype=np.float),
                           'inventory': [],
                           'inventory_vector': np.zeros(len(self.keys), dtype=np.float),
                           'final_goal': final_goal,
                           'final_goal_loc': final_goal_loc}
        if not act_test:
            return opt_observation, act_observation
        else:
            return act_observation

    def step(self, opt_obs, act_obs, action):
        """
        System steps with a primitive action.

        :return: New observations, rewards, terminal flags
        """
        act_observation = dcp(act_obs)
        act_observation_ = dcp(act_observation)
        s, ag, ag_loc, inv, inv_vec = self._move_agent(act_observation, action)
        act_observation_['state'], act_observation_['achieved_goal'] = s, ag
        act_observation_['achieved_goal_loc'] = ag_loc
        act_observation_['inventory'], act_observation_['inventory_vector'] = inv, inv_vec
        if act_observation['desired_goal'] == act_observation_['achieved_goal']:
            # option done when the low level agent achieves the opted-desired goal
            act_reward, opt_done = 1.0, True
        else:
            act_reward, opt_done = 0.0, False

        if opt_obs is None:
            return act_observation_, act_reward, opt_done
        else:
            # get option info
            opt_observation = dcp(opt_obs)
            opt_observation_ = dcp(opt_observation)
            opt_observation_['state'] = act_observation_['state'].copy()
            opt_observation_['inventory'] = act_observation_['inventory'].copy()
            opt_observation_['inventory_vector'] = act_observation_['inventory_vector'].copy()
            if opt_observation['final_goal'] == act_observation_['achieved_goal']:
                # episode done when the low level agent achieves the final goal
                opt_reward, ep_done = 1.0, True
            else:
                opt_reward, ep_done = 0.0, False
            return act_observation_, act_reward, ep_done, opt_observation_, opt_reward, opt_done

    def _move_agent(self, observation, action):
        """
        Move an agent with an primitive action.

        :return: New state, achieved goal, coordinate of the achieved goal, inventory, inventory one-hot vector
        """
        x, y = int(observation['state'][0]), int(observation['state'][1])
        x_, y_ = x, y
        if action == 0:
            # agent attempts to move up
            y_ = y+1
            if self.world_running["row"+str(y_)][x_] == 0:
                # encounter a wall
                y_ = y
        elif action == 1:
            # agent attempts to move down
            y_ = y-1
            if self.world_running["row"+str(y_)][x_] == 0:
                y_ = y
        elif action == 2:
            # agent attempts to move left
            x_ = x-1
            if self.world_running["row"+str(y_)][x_] == 0:
                x_ = x
        elif action == 3:
            # agent attempts to move right
            x_ = x+1
            if self.world_running["row"+str(y_)][x_] == 0:
                x_ = x
        else:
            raise ValueError("Action {} does not exist".format(action))
        # _check_move() return state_ and achieved_goal
        state_, ag, ag_loc, inv, inv_vector = self._check_move(x, y, x_, y_, observation['inventory'],
                                                               observation['inventory_vector'])
        return state_, ag, ag_loc, inv, inv_vector

    def _check_move(self, x, y, x_, y_, inventory, inventory_vector):
        """
        Check if a movement is valid, return an achieved goal and a possibly changed inventory

        :return: New state, achieved goal, coordinate of the achieved goal, inventory, inventory one-hot vector
        """
        achieved_goal = self.world_running["row" + str(y_)][x_]
        if achieved_goal == 1:
            state_ = np.array(([x_, y_]), dtype=np.float)
        # agent picks up a key
        elif achieved_goal in self.keys_running:
            state_ = np.array(([x_, y_]), dtype=np.float)
            inventory.append(dcp(achieved_goal))
            ind = self.keys.index(achieved_goal)
            if inventory_vector[ind] == 0:
                inventory_vector[ind] = 1
            else:
                raise ValueError("Something is wrong here")
            self.keys_running.remove(achieved_goal)
            self.world_running["row" + str(y_)][x_] = 1
        # agent attempts to open a door
        elif achieved_goal in self.doors:
            requested_key = dcp(achieved_goal)
            requested_key = requested_key[0]+'k'+requested_key[2]
            if requested_key not in inventory:
                # agent does not carry the corresponding key
                x_ = x
                y_ = y
                state_ = np.array(([x_, y_]), dtype=np.float)
                achieved_goal = self.world_running["row" + str(y_)][x_]
            else:
                # agent opens the door successfully
                state_ = np.array(([x_, y_]), dtype=np.float)
        # agent reaches a final goal
        elif achieved_goal in self.final_goals:
            state_ = np.array(([x_, y_]), dtype=np.float)
        else:
            state_ = None
            raise ValueError("Something is wrong here")
        achieved_goal_loc = np.array(([x_, y_]), dtype=np.float)
        return state_, achieved_goal, achieved_goal_loc, inventory, inventory_vector

    def get_goal_location(self, goal):
        """
        :return: The coordinate of a goal.
        """
        return np.array(([self.key_door_dict[goal][2], self.key_door_dict[goal][1]]), dtype=np.float)

    @staticmethod
    def _create_world(init_room_height, middle_room_size, middle_room_num, final_room_num):
        """
        This function automatically create a world given some information of the world.
            The height of the hall room is manual given.
            The height of the world is determined by the height of the hall room and the size of middle/final rooms.
            The width of the world is determined by the size and number of middle/final rooms.

        :param init_room_height:  The height of the hall room (beginning of every episode)
        :param middle_room_size:  The size of the middle and final rooms (they are the same, and they are square)
        :param middle_room_num:   The number of the middle rooms
        :param final_room_num:    The number of the final rooms
        :return: Dictionaries of the world and all of the goals.
        """
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
        key_door_dict = dict.fromkeys(["md"+str(i) for i in range(middle_room_num)] +
                                      ["mk"+str(i) for i in range(middle_room_num)] +
                                      ["fd"+str(i) for i in range(final_room_num)] +
                                      ["fk"+str(i) for i in range(final_room_num)] +
                                      ["fg"+str(i) for i in range(final_room_num)])

        # calculate the door coordinate for the first middle room
        middle_door_x = middle_room_size // 2
        mkls = []
        for _ in range(middle_room_num):
            # using 'mdi' to represent a door in the main room
            key_door_dict['md'+str(_)] = ['i', middle_room_size*2+2, middle_door_x+1*(_+1) + middle_room_size*_]
            world_str[middle_room_size*2+2][middle_door_x+1*(_+1) + middle_room_size*_] = 'md'+str(_)
            # randomly place the key for each middle room within the main room
            done = False
            while not done:
                mkl = (r.randint(0, init_room_height-1), r.randint(1, init_room_width-1))
                if mkl not in mkls:
                    mkls.append(mkl)
                    key_door_dict['mk' + str(_)] = ['i', mkl[0]+middle_room_size*2+3, mkl[1]+1]
                    world_str[mkl[0]+middle_room_size*2+3][mkl[1]+1] = 'mk' + str(_)
                    done = True
        # calculate the door coordinate for each final room within a middle room
        final_door_x = middle_room_size // 2
        fkls = []
        for _ in range(final_room_num):
            # using 'fdi' to represent a door of a final room
            key_door_dict['fd'+str(_)] = ['m',  middle_room_size+1,
                                          final_door_x+1*(block_room_num+_+1) + middle_room_size*(block_room_num+_)]
            world_str[middle_room_size+1][final_door_x+1*(block_room_num+_+1) + middle_room_size*(block_room_num+_)] \
                = 'fd' + str(_)
            fg_xy = (r.randint(0, middle_room_size-1), r.randint(0, middle_room_size-1))
            key_door_dict['fg'+str(_)] = ['f', fg_xy[0]+1,
                                          fg_xy[1]+1*(block_room_num+_+1) + middle_room_size*(block_room_num+_)]
            world_str[fg_xy[0]+1][fg_xy[1]+1*(block_room_num+_+1) + middle_room_size*(block_room_num+_)] = 'fg' + str(_)
            # randomly place the key of each final room within the main room or a middle room
            done = False
            while not done:
                fk_mr = r.randint(0, middle_room_num-1)
                fkl = (r.randint(0, middle_room_size-1), r.randint(0, middle_room_size-1))
                if fkl not in fkls:
                    fkls.append(fkl)
                    key_door_dict['fk'+str(_)] = ['m', fkl[0]+middle_room_size+2,
                                                  fkl[1]+1*(fk_mr+1) + middle_room_size*fk_mr]
                    world_str[fkl[0]+middle_room_size+2][fkl[1]+1*(fk_mr+1) + middle_room_size*fk_mr] = 'fk'+str(_)
                    done = True

        world_dict = dict.fromkeys(["row"+str(len(world_np)-i-1) for i in range(len(world_np))])
        for i in range(len(world_str)):
            world_dict["row"+str(i)] = world_str[-i-1]
        for n in key_door_dict.items():
            n[1][1] = len(world_dict) - n[1][1] - 1

        return world_dict, key_door_dict
