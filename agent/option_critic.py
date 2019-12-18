import os
import random as R
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from agent.utils.replay_buffer import ReplayBuffer
from agent.utils.networks import Critic, IntraPolicy
from agent.utils.exploration_strategy import ExpDecayGreedy


# TODO: add opt_learn(), opt_remember(), select_action(), select_option()
class OptionCritic(object):
    def __init__(self, env_params, opt_tr_namedtuple, path=None, seed=0,
                 option_lr=1e-5, opt_mem_capacity=int(1e6), opt_batch_size=256, opt_tau=0.2,
                 option_num=4, eps_start=1, eps_end=0.05, eps_decay=30000,
                 action_lr=1e-5,
                 gamma=0.98):
        T.manual_seed(seed)
        R.seed(seed)
        if path is None:
            self.ckpt_path = "ckpts"
        else:
            self.ckpt_path = path + "/ckpts"
        if not os.path.isdir(self.ckpt_path):
            os.mkdir(self.ckpt_path)
        use_cuda = T.cuda.is_available()
        self.device = T.device("cuda" if use_cuda else "cpu")

        self.option_num = option_num
        self.act_input_dim = env_params['act_input_dim']
        self.act_output_dim = env_params['act_output_dim']
        self.act_max = env_params['act_max']
        self.input_rescale = env_params['input_rescale']
        self.input_max = env_params['input_max']
        self.input_min = env_params['input_min']
        self.env_type = env_params['env_type']
        if self.env_type not in ['OR', 'TRE', 'TRH']:
            raise ValueError("Wrong environment type: {}, must be one of 'OR', 'TRE', 'TRH'".format(self.env_type))

        self.intra_policy = IntraPolicy(self.act_input_dim, self.option_num, self.act_output_dim)
        self.intra_policy_optimizer = Adam(self.intra_policy.parameters(), lr=action_lr)

        self.opt_exploration = ExpDecayGreedy(eps_start, eps_end, eps_decay)
        self.option_q = Critic(self.act_input_dim, self.option_num)
        self.option_q_target = Critic(self.act_input_dim, self.option_num)
        self.option_optimizer = Adam(self.option_q.parameters(), lr=option_lr)
        self.option_memory = ReplayBuffer(opt_mem_capacity, opt_tr_namedtuple, seed=seed)
        self.opt_batch_size = opt_batch_size
        self.opt_tau = opt_tau
        self.opt_soft_update(tau=1)

    def opt_soft_update(self, tau=None):
        if tau is None:
            tau = self.opt_tau
        for target_param, param in zip(self.option_q_target.parameters(), self.option_q.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def save_network(self, epo):
        T.save(self.option_q_target.state_dict(), self.ckpt_path+"/ckpt_option_q_target_epo"+str(epo)+".pt")
        T.save(self.intra_policy.state_dict(), self.ckpt_path+"/ckpt_intra_policy_epo"+str(epo)+".pt")

    def load_network(self, epo):
        self.option_q_target.load_state_dict(T.load(self.ckpt_path+'/ckpt_option_q_target_epo'+str(epo)+'.pt'))
        self.intra_policy.load_state_dict(T.load(self.ckpt_path+'/ckpt_intra_policy_epo'+str(epo)+'.pt'))

    def normalize(self, inputs):
        return self.input_rescale*((inputs - self.input_min) / (self.input_max - self.input_min))