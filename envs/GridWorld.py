import numpy as np
import random as r
from copy import deepcopy as dcp


class GridWorldEnv(object):
    """
    --Brief Introduction:
        An grid-world constituted by two types of rooms: a hall and some locked rooms.
        When calling function reset(), an agent will be initialized at a random cell within the hall.
        In order to enter a locked rooms, an agent needs to first collect the corresponding keys.
        Keys, doors and goals are placed at where you want them to be.

    --MDP Definition:
        States and goals are represented by [x, y] coordinates of the cells.
        Observation is constituted by a state, a goal and an inventory of keys.
            The inventory is a one-hot vector with size of the number of keys in an environment instance.
        Primitive actions are "up", "down", "left" and "right" (deterministic).
            Reward for primitive-agent is 1 when a desired goal given by the option-agent is achieved and 0 otherwise.
        Options are coordinates of goals.
            Reward for option-agent is 1 when an ultimate goal is achieved and 0 otherwise.
    """
    def __init__(self, setup, seed=2222):
        r.seed(seed)
        self.env_type = ""
        self.random_key = True
        self.env_setup = setup
        self.world, self.key_door_dict = self._create_world(self.env_setup)
        self.init_width = self.env_setup['init_width']
        self.init_height = self.env_setup['init_height']
        self.keys = [key for key in self.key_door_dict if "k" in key]
        self.doors = [key for key in self.key_door_dict if "d" in key]
        self.final_goals = [key for key in self.key_door_dict if "fg" in key]
        self.goal_space = self.keys + self.doors + self.final_goals
        self.option_space = [op for op in range(len(self.goal_space))]
        self.action_space = [0, 1, 2, 3]
        self.actions = ['up', 'down', 'left', 'right']
        self.keys_running = None
        self.world_running = None
        self.input_max = np.array(
            ([len(self.world['row0'])-2, len(self.world)-2, len(self.world['row0'])-2, len(self.world)-2]
             + [1 for k in range(len(self.keys))]), dtype=np.float
        )
        self.input_min = np.array(
            ([1, 1, 1, 1]+[0 for k in range(len(self.keys))]), dtype=np.float
        )

    def reset(self, act_test=False):
        """
        Every time an episode ends, call this function.

        :param act_test:  If True, this function only return low level observations
                          This is for testing low-level policy, or non-hierarchical agent.
        :return:          Initial observations
        """
        if self.random_key:
            self._reset_keys()
        self.world_running = dcp(self.world)
        self.keys_running = dcp(self.keys)
        x, y = r.randint(1, self.init_width), r.randint(1, self.init_height)
        final_goal = r.choice(self.final_goals)
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
            return act_observation

    def step(self, opt_obs, act_obs, action):
        """
        System steps with a primitive action.

        :return: New observations, rewards, terminal flags
        """
        act_observation = dcp(act_obs)
        act_observation_ = dcp(act_observation)
        state_, achieved_goal, achieved_goal_loc, inventory, inventory_vector = self._move_agent(act_observation, action)
        act_observation_['state'] = state_
        act_observation_['achieved_goal'] = achieved_goal
        act_observation_['achieved_goal_loc'] = achieved_goal_loc
        act_observation_['inventory'] = inventory
        act_observation_['inventory_vector'] = inventory_vector
        if act_observation_['desired_goal'] == act_observation_['achieved_goal']:
            # option done when the low level agent achieves the opted-desired goal
            act_reward, opt_done = 1.0, True
        else:
            act_reward, opt_done = 0.0, False

        if opt_obs is None:
            # When the option observation is None, this function works for non-hierarchical RL agent
            return act_observation_, act_reward, opt_done
        else:
            # get option info
            opt_observation_ = dcp(opt_obs)
            opt_observation_['state'] = dcp(act_observation_['state'])
            opt_observation_['inventory'] = dcp(act_observation_['inventory'])
            opt_observation_['inventory_vector'] = dcp(act_observation_['inventory_vector'])
            if opt_observation_['final_goal'] == act_observation_['achieved_goal']:
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
        return self._check_move(x, y, x_, y_, observation['inventory'], observation['inventory_vector'])

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
            ind = self.keys.index(achieved_goal)
            state_ = np.array(([x_, y_]), dtype=np.float)
            if achieved_goal not in inventory:
                inventory.append(dcp(achieved_goal))
                inventory_vector[ind] = 1

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
        """This function is used to create a world given some setup info.
        """
        raise NotImplementedError()

    def _reset_keys(self):
        """This function is only for random key grid world
        """
        pass