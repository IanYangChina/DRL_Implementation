import numpy as np
import random as r
from copy import deepcopy as dcp


class GridWorldEnv(object):
    """
    --Brief Introduction:
        An grid-world constituted by two types of rooms: a hall and some locked rooms.
        When calling function reset(), an agent will be initialized at a random cell within the hall.
        In order to enter a locked rooms, an agent needs to first collect the corresponding keys.
        Keys are randomly placed at where you want them to be.
        The ultimate goal-locations are cells within locked rooms.

    --MDP Definition:
        States and goals are represented by [x, y] coordinates of the cells.
        Observation is constituted by a state, a goal and an inventory of keys.
            The inventory is a one-hot vector whose size is the number of keys in an environment instance.
        Primitive actions are "up", "down", "left" and "right" (deterministic actions).
            Reward for primitive-agent is 1 when a desired goal given by the option-agent is achieved and 0 otherwise.
        Options are coordinates of goals.
            Reward for option-agent is 1 when an ultimate goal is achieved and 0 otherwise.
    """
    def __init__(self, setup, seed=2222):
        r.seed(seed)
        self.world, self.key_door_dict = self._create_world(setup)
        self.init_width = setup['init_width']
        self.init_height = setup['init_height']
        self.keys = [key for key in self.key_door_dict if "k" in key]
        self.doors = [key for key in self.key_door_dict if "d" in key]
        self.final_goals = [key for key in self.key_door_dict if "fg" in key]
        self.goal_space = self.keys + self.doors + self.final_goals
        self.option_space = [op for op in range(len(self.goal_space))]
        self.action_space = [0, 1, 2, 3]
        self.actions = ['up', 'down', 'left', 'right']
        self.keys_running = None
        self.world_running = None

        self.input_max = np.array(([len(self.world['row0']), len(self.world)-1,
                                    len(self.world['row0']), len(self.world)-1]
                                   + [1 for k in range(len(self.keys))]),
                                  dtype=np.float)
        self.input_min = np.array(([1, 1, 1, 1]+[0 for k in range(len(self.keys))]), dtype=np.float)

    def reset(self, all_goal=False, act_test=False):
        """
        Every time an episode ends, call this function.

        :param all_goal:  If True, all of the goals are possible to be the high level agent's desired goal.
        :param act_test:  If True, this function only return low level observations (used for test the low level agent)
        :return:          Initial observations
        """
        self.world_running = dcp(self.world)
        self.keys_running = dcp(self.keys)
        x, y = r.randint(1, self.init_width), r.randint(1, self.init_height)
        if not all_goal:
            final_goal = r.choice(self.final_goals)
        else:
            final_goal = r.choice(self.final_goals + self.doors + self.keys)

        final_goal_loc = np.array(([self.key_door_dict[final_goal][1], self.key_door_dict[final_goal][0]]), dtype=np.float)
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
            act_observation['desired_goal'] = r.choice(self.goal_space)
            act_observation['desired_goal_loc'] = self.get_goal_location(act_observation['desired_goal'])
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
            try:
                requested_key = requested_key[-3] + 'k' + requested_key[-1]
            except:
                requested_key = 'k' + requested_key[-1]
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
        return np.array(([self.key_door_dict[goal][1], self.key_door_dict[goal][0]]), dtype=np.float)

    def _create_world(self, setup):
        """This function automatically create a world given some information of the world.
        """
        raise NotImplementedError()