import os
import random as R
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from collections import namedtuple
from ..utils.networks import Critic
from ..utils.replay_buffer import ReplayBuffer, GridWorldHindsightReplayBuffer
from ..utils.exploration_strategy import ExpDecayGreedy, GoalSucRateBasedExpGreed
OptTr = namedtuple('OptionTransition',
                   ('state', 'inventory', 'final_goal', 'option', 'next_state', 'next_inventory', 'next_goal',
                    'option_done', 'reward', 'done'))
ActTr = namedtuple('ActionTransition',
                   ('state', 'inventory', 'desired_goal', 'action', 'next_state', 'next_inventory', 'next_goal',
                    'achieved_goal', 'reward', 'done'))


class OptionDQN(object):
    def __init__(self, env_params, path=None, seed=0,
                 act_exploration=None, sub_suc_percentage=None, gsrb_decay=None, act_eps_decay=30000,
                 option_lr=1e-5, opt_mem_capacity=int(1e6), opt_batch_size=256, opt_tau=0.2,
                 action_lr=1e-5, act_mem_capacity=int(1e6), act_batch_size=256, act_tau=0.05, clip_value=5.0,
                 optimization_steps=2, gamma=0.99, eps_start=1, eps_end=0.05, eps_decay=30000,
                 opt_eps_decay_start=None):
        T.manual_seed(seed)
        R.seed(seed)
        if path is None:
            self.ckpt_path = "ckpts"
        else:
            self.ckpt_path = path+"/ckpts"
        if not os.path.isdir(self.ckpt_path):
            os.mkdir(self.ckpt_path)
        use_cuda = T.cuda.is_available()
        self.device = T.device("cuda" if use_cuda else "cpu")

        self.opt_input_dim = env_params['opt_input_dim']
        self.opt_output_dim = env_params['opt_output_dim']
        self.opt_max = env_params['opt_max']
        self.act_input_dim = env_params['act_input_dim']
        self.act_output_dim = env_params['act_output_dim']
        self.act_max = env_params['act_max']
        self.input_rescale = env_params['input_rescale']
        self.input_max = env_params['input_max']
        self.input_min = env_params['input_min']
        self.env_type = env_params['env_type']
        if self.env_type not in ['OR', 'TRE', 'TRH']:
            raise ValueError("Wrong environment type: {}, must be one of 'OR', 'TRE', 'TRH'".format(self.env_type))

        self.opt_exploration = ExpDecayGreedy(eps_start, eps_end, eps_decay, opt_eps_decay_start)
        self.option_agent = Critic(self.opt_input_dim, self.opt_output_dim).to(self.device)
        self.option_target = Critic(self.opt_input_dim, self.opt_output_dim).to(self.device)
        self.option_optimizer = Adam(self.option_agent.parameters(), lr=option_lr)
        self.option_memory = ReplayBuffer(opt_mem_capacity, OptTr, seed=seed)
        self.opt_batch_size = opt_batch_size
        self.opt_mean_q_tmp = []
        self.opt_mean_q = []

        if act_exploration is None:
            self.act_exploration = ExpDecayGreedy(eps_start, eps_end, decay=act_eps_decay)
        elif act_exploration == 'GSRB':
            self.act_exploration = GoalSucRateBasedExpGreed(goals=env_params['goals'],
                                                            sub_suc_percentage=sub_suc_percentage, decay=gsrb_decay)
        else:
            raise ValueError("Specify wrong type of exploration strategy: {}".format(act_exploration))
        self.action_agent = Critic(self.act_input_dim, self.act_output_dim).to(self.device)
        self.action_target = Critic(self.act_input_dim, self.act_output_dim).to(self.device)
        self.action_optimizer = Adam(self.action_agent.parameters(), lr=action_lr)
        self.action_memory = GridWorldHindsightReplayBuffer(act_mem_capacity, ActTr, seed=seed)
        self.act_batch_size = act_batch_size
        self.act_mean_q_tmp = []
        self.act_mean_q = []

        self.gamma = gamma
        self.clip_value = clip_value
        self.opt_tau = opt_tau
        self.act_tau = act_tau
        self.opt_soft_update(tau=1)
        self.act_soft_update(tau=1)
        self.optimization_steps = optimization_steps

    def select_option(self, opt_obs, ep=None):
        inputs = np.concatenate((opt_obs['state'], opt_obs['final_goal_loc'], opt_obs['inventory_vector']), axis=0)
        inputs = self.normalize(inputs)
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        self.option_target.eval()
        option_values = self.option_target(inputs)
        if ep is None:
            option = T.argmax(option_values).item()
            return option
        else:
            _ = R.uniform(0, 1)
            if _ < self.opt_exploration(ep):
                option = R.randint(0, self.opt_max)
            else:
                option = T.argmax(option_values).item()
            return option

    def select_action(self, act_obs, ep=None):
        inputs = np.concatenate((act_obs['state'], act_obs['desired_goal_loc'], act_obs['inventory_vector']), axis=0)
        inputs = self.normalize(inputs)
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        self.action_target.eval()
        action_values = self.action_target(inputs)
        if ep is None:
            action = T.argmax(action_values).item()
            return action
        else:
            _ = R.uniform(0, 1)
            if isinstance(self.act_exploration, GoalSucRateBasedExpGreed):
                eps = self.act_exploration(act_obs['desired_goal'], ep)
            elif isinstance(self.act_exploration, ExpDecayGreedy):
                eps = self.act_exploration(ep)

            if _ < eps:
                action = R.randint(0, self.act_max)
            else:
                action = T.argmax(action_values).item()
            return action

    def remember(self, new, level, *args):
        if level == "option":
            self.option_memory.store_experience(new, *args)
        elif level == "action":
            self.action_memory.store_experience(new, *args)
        else:
            raise ValueError("Storing experience for wrong level {} has been requested".format(level))

    def learn(self, level, steps=None, batch_size=None, hindsight=True):
        if steps is None:
            steps = self.optimization_steps

        if level == "option":
            self.option_memory.store_episodes()
            if batch_size is None:
                batch_size = self.opt_batch_size
            for s in range(steps):
                self.opt_learn(batch_size)
                self.opt_soft_update()
        elif level == "action":
            if hindsight:
                self.action_memory.modify_episodes()
            self.action_memory.store_episodes()
            if batch_size is None:
                batch_size = self.act_batch_size
            for s in range(steps):
                self.act_learn(batch_size)
                self.act_soft_update()
        else:
            raise ValueError("Learning of a wrong level {} has been requested".format(level))

    def opt_learn(self, batch_size=None):
        if batch_size is None:
            batch_size = self.opt_batch_size
        if len(self.option_memory) < (2*batch_size):
            return
        batch = self.option_memory.sample(batch_size)

        inputs = np.concatenate((batch.state, batch.final_goal, batch.inventory), axis=1)
        inputs_ = np.concatenate((batch.next_state, batch.next_goal, batch.next_inventory), axis=1)
        inputs = self.normalize(inputs)
        inputs_ = self.normalize(inputs_)
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        inputs_ = T.tensor(inputs_, dtype=T.float).to(self.device)

        options = T.tensor(batch.option, dtype=T.long).unsqueeze(1).to(self.device)
        rewards = T.tensor(batch.reward, dtype=T.float).unsqueeze(1).to(self.device)
        option_done = T.tensor(batch.option_done, dtype=T.float).unsqueeze(1).to(self.device)
        episode_done = T.tensor(batch.done, dtype=T.float).unsqueeze(1).to(self.device)

        self.option_agent.train()
        self.option_target.eval()
        estimated_option_values = self.option_agent(inputs)
        self.opt_mean_q_tmp.append(np.mean(estimated_option_values.cpu().detach().numpy()))
        estimated_option_values = estimated_option_values.gather(1, options)
        unchanged_next_option_values = self.option_target(inputs_).gather(1, options)
        maximal_next_option_values = self.option_target(inputs_).max(1)[0].view(batch_size, 1)
        next_option_values = option_done * unchanged_next_option_values + (1 - option_done) * maximal_next_option_values
        target_option_values = rewards + episode_done*self.gamma*next_option_values
        target_option_values = T.clamp(target_option_values, 0.0, self.clip_value)

        self.option_optimizer.zero_grad()
        loss = F.smooth_l1_loss(estimated_option_values, target_option_values)

        loss.backward()
        self.option_optimizer.step()

        self.option_agent.eval()

    def act_learn(self, batch_size=None):
        if batch_size is None:
            batch_size = self.act_batch_size
        if len(self.action_memory) < (2*batch_size):
            return
        batch = self.action_memory.sample(batch_size)
        inputs = np.concatenate((batch.state, batch.desired_goal, batch.inventory), axis=1)
        inputs_ = np.concatenate((batch.next_state, batch.next_goal, batch.next_inventory), axis=1)
        inputs = self.normalize(inputs)
        inputs_ = self.normalize(inputs_)
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        inputs_ = T.tensor(inputs_, dtype=T.float).to(self.device)

        actions = T.tensor(batch.action, dtype=T.long).unsqueeze(1).to(self.device)
        rewards = T.tensor(batch.reward, dtype=T.float).unsqueeze(1).to(self.device)
        episode_done = T.tensor(batch.done, dtype=T.float).unsqueeze(1).to(self.device)

        self.action_agent.train()
        self.action_target.eval()
        estimated_action_values = self.action_agent(inputs)
        self.act_mean_q_tmp.append(np.mean(estimated_action_values.cpu().detach().numpy()))
        estimated_action_values = estimated_action_values.gather(1, actions)
        maximal_next_action_values = self.action_target(inputs_).max(1)[0].view(batch_size, 1)
        target_action_values = rewards + episode_done*self.gamma*maximal_next_action_values
        target_action_values = T.clamp(target_action_values, 0.0, self.clip_value)

        self.action_optimizer.zero_grad()
        loss = F.smooth_l1_loss(estimated_action_values, target_action_values)
        loss.backward()
        self.action_optimizer.step()

        self.action_agent.eval()

    def opt_soft_update(self, tau=None):
        if tau is None:
            tau = self.opt_tau
        for target_param, param in zip(self.option_target.parameters(), self.option_agent.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def act_soft_update(self, tau=None):
        if tau is None:
            tau = self.act_tau
        for target_param, param in zip(self.action_target.parameters(), self.action_agent.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def save_network(self, epo):
        T.save(self.option_target.state_dict(), self.ckpt_path+"/ckpt_option_target_epo"+str(epo)+".pt")
        T.save(self.action_target.state_dict(), self.ckpt_path+"/ckpt_action_target_epo"+str(epo)+".pt")

    def load_network(self, epo):
        self.option_agent.load_state_dict(T.load(self.ckpt_path+'/ckpt_option_target_epo' + str(epo) + '.pt'))
        self.action_agent.load_state_dict(T.load(self.ckpt_path+'/ckpt_action_target_epo' + str(epo) + '.pt'))
        self.option_target.load_state_dict(T.load(self.ckpt_path+'/ckpt_option_target_epo'+str(epo)+'.pt'))
        self.action_target.load_state_dict(T.load(self.ckpt_path+'/ckpt_action_target_epo'+str(epo)+'.pt'))

    def normalize(self, inputs):
        return self.input_rescale*((inputs - self.input_min) / (self.input_max - self.input_min))
