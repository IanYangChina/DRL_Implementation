import os
import numpy as np
from plot import smoothed_plot
from copy import deepcopy as dcp
from collections import namedtuple
from envs.GridWorld_TwoRoomTypes import TwoRoomType
from agent.herdqn_option_discrete import OptionDQN
path = os.path.dirname(os.path.realpath(__file__))

OptTr = namedtuple('OptionTransition',
                   ('state', 'inventory', 'final_goal', 'option', 'next_state', 'next_inventory', 'next_goal',
                    'option_done', 'reward', 'done'))
ActTr = namedtuple('ActionTransition',
                   ('state', 'inventory', 'desired_goal', 'action', 'next_state', 'next_inventory', 'next_goal',
                    'achieved_goal', 'reward', 'done'))
env_setup = {'middle_room_size': 5,
             'middle_room_num': 3,
             'final_room_num': 3,
             'main_room_height': 20}
env = TwoRoomType(env_setup, seed=2222)
# Print world layout and sub-goals (options)
for key in env.world:
    print(key, env.world[key])
print(env.option_space)
print(env.goal_space)
opt_obs, act_obs = env.reset()
# Training setup
EPOCH = 301
CYCLE = 50
EPISODE = 16
OPT_EPISODE_T = 50*len(env.final_goals)
ACT_EPISODE_T = 50*len(env.goal_space)
TIMESTEP = 70
TIMESTEP_T = 70
# Instantiate an agent
env_params = {'input_max': env.input_max,
              'input_min': env.input_min,
              'opt_input_dim': opt_obs['state'].shape[0]+opt_obs['final_goal_loc'].shape[0]+opt_obs['inventory_vector'].shape[0],
              'opt_output_dim': len(env.option_space),
              'opt_max': np.max(env.option_space),
              'act_input_dim': act_obs['state'].shape[0]+act_obs['desired_goal_loc'].shape[0]+act_obs['inventory_vector'].shape[0],
              'act_output_dim': len(env.action_space),
              'act_max': np.max(env.action_space),
              }
agent = OptionDQN(env_params, OptTr, ActTr, path=path, opt_eps_decay_start=EPOCH*CYCLE*EPISODE*0.5)
test = False
test_success_rates = []
opt_success_rates = []
act_success_rates = []
act_test_sus_rates = []
for epo in range(EPOCH):
    for cyc in range(CYCLE):
        if test:
            break
        opt_sus = 0
        act_sus = 0
        opt_ep_steps = 0
        for ep in range(EPISODE):
            opt_ep_returns = 0
            act_ep_returns = 0
            ep_time_step = 0
            ep_done = False
            opt_obs, act_obs = env.reset()
            while (not ep_done) and (ep_time_step < TIMESTEP):
                opt_done = False
                new_option = True
                opt_ep_steps += 1
                option = agent.select_option(opt_obs, ep=((ep+1)*(cyc+1)*(epo+1)))
                act_obs['desired_goal'] = env.goal_space[option]
                act_obs['desired_goal_loc'] = env.get_goal_location(act_obs['desired_goal'])
                # low level process starts
                while (not opt_done) and (ep_time_step < TIMESTEP):
                    ep_time_step += 1
                    # action = agent.select_action(act_obs, ep=None)
                    action = agent.select_action(act_obs, ep=((ep+1)*(cyc+1)*(epo+1)))
                    act_obs_, act_reward, ep_done, opt_obs_, opt_reward, opt_done = env.step(opt_obs, act_obs, action)
                    act_obs['achieved_goal_loc'] = dcp(act_obs_['achieved_goal_loc'])
                    opt_ep_returns += opt_reward
                    act_ep_returns += act_reward
                    # store transitions and renew observation
                    agent.remember(new_option, "action",
                                   act_obs['state'], act_obs['inventory_vector'], act_obs['desired_goal_loc'], action,
                                   act_obs_['state'], act_obs_['inventory_vector'], act_obs_['desired_goal_loc'],
                                   act_obs['achieved_goal_loc'], act_reward, 1-int(opt_done))
                    act_obs = act_obs_.copy()
                    agent.remember(new_option, "option",
                                   opt_obs['state'], opt_obs['inventory_vector'], opt_obs['final_goal_loc'], option,
                                   opt_obs_['state'], opt_obs_['inventory_vector'], opt_obs_['final_goal_loc'],
                                   1-int(opt_done), opt_reward, 1-int(ep_done))
                    opt_obs = opt_obs_.copy()
                    new_option = False
            opt_sus += opt_ep_returns
            act_sus += act_ep_returns
            agent.apply_hindsight(hindsight=True)
            agent.learn("action", steps=4)
        agent.learn("option", steps=4)
        opt_success_rates.append(opt_sus/EPISODE)
        act_success_rates.append(act_sus/opt_ep_steps)
        print("Epoch %i" % epo, "Cycle %i" % cyc,
              "Option SucRate {}/{}".format(int(opt_sus), EPISODE),
              "Action SucRate {}/{}".format(int(act_sus), opt_ep_steps))

    if (epo % 1 == 0) and (epo != 0):
        """Testing primitive agent"""
        goal_num = len(env.goal_space)
        success_t = [env.goal_space,
                     [0 for g in range(goal_num)],
                     [0 for g in range(goal_num)]]
        goal_ind_t = 0
        for ep_t in range(ACT_EPISODE_T):
            act_obs_t = env.reset(act_test=True)
            act_obs_t['desired_goal'] = env.goal_space[goal_ind_t]
            act_obs_t['desired_goal_loc'] = env.get_goal_location(act_obs_t['desired_goal'])
            success_t[2][goal_ind_t] += 1
            opt_done_t = False
            ep_time_step_t = 0
            while (not opt_done_t) and (ep_time_step_t < TIMESTEP_T):
                ep_time_step_t += 1
                action_t = agent.select_action(act_obs_t)
                act_obs_t_, act_reward_t, opt_done_t = env.step(None, act_obs_t, action_t)
                success_t[1][goal_ind_t] += int(act_reward_t)
                act_obs_t['achieved_goal'] = dcp(act_obs_t_['achieved_goal'])
                act_obs_t['achieved_goal_loc'] = dcp(act_obs_t_['achieved_goal_loc'])
                act_obs_t = act_obs_t_.copy()
            goal_ind_t = (goal_ind_t+1) % goal_num
        act_test_sus_rates.append(sum(success_t[1])/sum(success_t[2]))
        print("Primitive agent test result:\n", success_t)

        """Testing both agents"""
        opt_goal_num = len(env.final_goals)
        opt_success_t = [env.final_goals,
                         [0 for g in range(opt_goal_num)],
                         [0 for g in range(opt_goal_num)]]
        opt_goal_ind_t = 0
        for ep_t in range(OPT_EPISODE_T):
            ep_done_t = False
            ep_time_step_t = 0
            opt_obs_t, act_obs_t = env.reset()
            opt_obs_t['final_goal'] = env.final_goals[opt_goal_ind_t]
            opt_obs_t['final_goal_loc'] = env.get_goal_location(opt_obs_t['final_goal'])
            opt_success_t[2][opt_goal_ind_t] += 1
            # print("\nNew Episode, ultimate goal: {}".format(opt_obs_t['final_goal']))
            while (not ep_done_t) and (ep_time_step_t < TIMESTEP_T):
                option = agent.select_option(opt_obs_t, ep=None)
                act_obs_t['desired_goal'] = env.goal_space[option]
                act_obs_t['desired_goal_loc'] = env.get_goal_location(act_obs_t['desired_goal'])
                opt_done_t = False
                # print("Option/subgoal: {}".format(act_obs_t['desired_goal']))
                while (not opt_done_t) and (ep_time_step_t < TIMESTEP_T):
                    ep_time_step_t += 1
                    action_t = agent.select_action(act_obs_t, ep=None)
                    act_obs_t_, act_reward_t, ep_done_t, opt_obs_t_, opt_reward_t, opt_done_t = \
                        env.step(opt_obs_t, act_obs_t, action_t)
                    opt_success_t[1][opt_goal_ind_t] += int(opt_reward_t)
                    act_obs_t['achieved_goal'] = dcp(act_obs_t_['achieved_goal'])
                    act_obs_t['achieved_goal_loc'] = dcp(act_obs_t_['achieved_goal_loc'])
                    # print("State: {}, action: {}, achieved goal: {}".format(act_obs_t['state'],
                    #                                                         env.actions[action_t],
                    #                                                         act_obs_t['achieved_goal']))
                    act_obs_t = act_obs_t_.copy()
                    opt_obs_t = opt_obs_t_.copy()
            opt_goal_ind_t = (opt_goal_ind_t + 1) % opt_goal_num
        test_success_rates.append(sum(opt_success_t[1])/sum(opt_success_t[2]))
        print("Option agent test result:\n", opt_success_t)

smoothed_plot("Success_rate_train_option.png", opt_success_rates, x_label="Cycle")
smoothed_plot("Success_rate_train_action.png", act_success_rates, x_label="Cycle")
smoothed_plot("Success_rate_test_option.png", test_success_rates, x_label="Epo")
smoothed_plot("Success_rate_test_action.png", act_test_sus_rates, x_label="Epo")
