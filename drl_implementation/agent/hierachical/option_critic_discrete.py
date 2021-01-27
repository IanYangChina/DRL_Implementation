import os
import random as R
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from ..utils.replay_buffer import ReplayBuffer
from ..utils.networks import Critic, IntraPolicy
from ..utils.exploration_strategy import ExpDecayGreedy
from collections import namedtuple
OptTr = namedtuple('OptionTransition',
                   ('state', 'inventory', 'desired_goal',
                    'option', 'achieved_goal', 'reward', 'done',
                    'next_state', 'next_inventory', 'next_goal'))


class OptionCritic(object):
    def __init__(self, env_params, path=None, seed=0,
                 option_lr=1e-5, opt_mem_capacity=int(1e6), opt_batch_size=128, opt_tau=0.5, opt_optimization_steps=10,
                 option_num=4, eps_start=1, eps_end=0.05, eps_decay=50000,
                 action_lr=1e-5, action_entropy_beta=0.01,
                 gamma=0.98, clip_value=5.0):
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
        self.input_dim = env_params['input_dim']
        self.primitive_output_dim = env_params['primitive_output_dim']
        self.input_rescale = env_params['input_rescale']
        self.input_max = env_params['input_max']
        self.input_min = env_params['input_min']
        self.env_type = env_params['env_type']
        if self.env_type not in ['OR', 'TRE', 'TRH']:
            raise ValueError("Wrong environment type: {}, must be one of 'OR', 'TRE', 'TRH'".format(self.env_type))

        self.intra_policies = []
        self.intra_policy_optimizers = []
        self.intra_critics = []
        self.intra_critic_optimizers = []
        for n in range(self.option_num):
            # [pi(s,o), beta(s)] * num_o
            self.intra_policies.append(IntraPolicy(self.input_dim, self.primitive_output_dim).to(self.device))
            self.intra_policy_optimizers.append(Adam(self.intra_policies[n].parameters(), lr=action_lr))
            # Q(s,o) * num_o (Q_U)
            self.intra_critics.append(Critic(self.input_dim, self.primitive_output_dim).to(self.device))
            self.intra_critic_optimizers.append(Adam(self.intra_critics[n].parameters(), lr=action_lr))
        self.action_entropy_beta = action_entropy_beta
        self.intra_policy_loss = []
        self.termination_loss = []

        # Q(s) over options (Q_omega)
        self.option_q = Critic(self.input_dim, self.option_num).to(self.device)
        self.option_q_target = Critic(self.input_dim, self.option_num).to(self.device)
        self.opt_exploration = ExpDecayGreedy(eps_start, eps_end, eps_decay)
        self.option_optimizer = Adam(self.option_q.parameters(), lr=option_lr)
        self.opt_optimization_steps = opt_optimization_steps
        self.option_policy_loss = []
        self.option_memory = ReplayBuffer(opt_mem_capacity, OptTr, seed=seed)
        self.opt_batch_size = opt_batch_size
        self.opt_tau = opt_tau
        self.opt_soft_update(tau=1)
        self.clip_value = clip_value
        self.gamma = gamma

    def select_option(self, opt_obs, ep=None):
        inputs = np.concatenate((opt_obs['state'], opt_obs['desired_goal_loc'], opt_obs['inventory_vector']), axis=0)
        inputs = self.normalize(inputs)
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        self.option_q_target.eval()
        option_values = self.option_q_target(inputs)
        if ep is None:
            option = T.argmax(option_values).item()
            return option
        else:
            _ = R.uniform(0, 1)
            if _ < self.opt_exploration(ep):
                option = R.randint(0, self.option_num-1)
            else:
                option = T.argmax(option_values).item()
            return option

    def select_action(self, option, act_obs):
        inputs = np.concatenate((act_obs['state'], act_obs['desired_goal_loc'], act_obs['inventory_vector']), axis=0)
        inputs = self.normalize(inputs)
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        self.intra_policies[option].eval()
        termination_probs, action_probs = self.intra_policies[option](inputs)
        termination_probs = termination_probs.detach().cpu().numpy()
        termination = np.random.choice([True, False], 1, p=[termination_probs[0], 1-termination_probs[0]])
        action_probs = action_probs.detach().cpu().numpy()
        action = np.random.choice(self.primitive_output_dim, 1, p=action_probs)
        return bool(termination), int(action)

    def intra_policy_learn(self, act_obs, act_obs_, option, action, reward, done):
        inputs = np.concatenate((act_obs['state'], act_obs['desired_goal_loc'], act_obs['inventory_vector']), axis=0)
        inputs = self.normalize(inputs)
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        inputs_ = np.concatenate((act_obs_['state'], act_obs_['desired_goal_loc'], act_obs_['inventory_vector']), axis=0)
        inputs_ = self.normalize(inputs_)
        inputs_ = T.tensor(inputs_, dtype=T.float).to(self.device)

        # Update intra_policy_critic
        self.option_q_target.eval()
        self.intra_policies[option].eval()
        self.intra_critics[option].train()
        estimated_Q_U_swa = self.intra_critics[option](inputs)[action].unsqueeze(0)
        termination_, _ = self.intra_policies[option](inputs_)
        estimated_U_ = (1-termination_)*self.option_q_target(inputs_)[option] + termination_*self.option_q_target(inputs_).max()
        target_Q_U_swa = reward + (1-done)*self.gamma*estimated_U_
        self.intra_critic_optimizers[option].zero_grad()
        intra_critic_loss = F.smooth_l1_loss(estimated_Q_U_swa, target_Q_U_swa)
        intra_critic_loss.backward()
        self.intra_critic_optimizers[option].step()
        self.intra_critics[option].eval()
        self.intra_policy_loss.append(intra_critic_loss.detach().cpu().numpy())

        # Update intra_policy
        self.intra_policies[option].train()
        termination, action_probs = self.intra_policies[option](inputs)
        entropy = -T.sum(action_probs*T.log(action_probs))
        # Subtracting baseline Q_U(s, o, a) - Q_Omega(s, o)
        Q_U_swa = self.intra_critics[option](inputs)[action]
        Q_U_swa = Q_U_swa - self.option_q_target(inputs)[option]
        self.intra_policy_optimizers[option].zero_grad()
        intra_policy_loss = -T.log(action_probs[action])*Q_U_swa - self.action_entropy_beta*entropy
        intra_policy_loss.backward(create_graph=True)

        Q_Omega = self.option_q_target(inputs)
        A_so = Q_Omega[option] - Q_Omega.max()
        termination_loss = termination*A_so
        termination_loss.backward()
        self.intra_policy_optimizers[option].step()
        self.intra_policies[option].eval()
        self.termination_loss.append(termination_loss.detach().cpu().numpy())

    def opt_remember(self, new, *args):
        self.option_memory.store_experience(new, *args)

    def opt_learn(self, batch_size=None, steps=None):
        if batch_size is None:
            batch_size = self.opt_batch_size
        self.option_memory.store_episodes()
        if len(self.option_memory) < (2*batch_size):
            return
        if steps is None:
            steps = self.opt_optimization_steps
        option_policy_loss_tmp = []
        for _ in range(steps):
            batch = self.option_memory.sample(batch_size)

            inputs = np.concatenate((batch.state, batch.desired_goal, batch.inventory), axis=1)
            inputs_ = np.concatenate((batch.next_state, batch.next_goal, batch.next_inventory), axis=1)
            inputs = self.normalize(inputs)
            inputs_ = self.normalize(inputs_)
            inputs = T.tensor(inputs, dtype=T.float).to(self.device)
            inputs_ = T.tensor(inputs_, dtype=T.float).to(self.device)

            options = T.tensor(batch.option, dtype=T.long).unsqueeze(1).to(self.device)
            rewards = T.tensor(batch.reward, dtype=T.float).unsqueeze(1).to(self.device)
            done = T.tensor(batch.done, dtype=T.float).unsqueeze(1).to(self.device)

            self.option_q.train()
            self.option_q_target.eval()
            estimated_values = self.option_q(inputs).gather(1, options)
            maximal_next_values = self.option_q_target(inputs_).max(1)[0].view(batch_size, 1)
            target_values = rewards + done*self.gamma*maximal_next_values
            target_values = T.clamp(target_values, 0.0, self.clip_value)

            self.option_optimizer.zero_grad()
            loss = F.smooth_l1_loss(estimated_values, target_values)
            loss.backward()
            self.option_optimizer.step()
            self.option_q.eval()
            option_policy_loss_tmp.append(loss.cpu().detach().numpy())
        self.option_policy_loss.append(np.mean(option_policy_loss_tmp))

    def opt_soft_update(self, tau=None):
        if tau is None:
            tau = self.opt_tau
        for target_param, param in zip(self.option_q_target.parameters(), self.option_q.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def save_network(self, epo):
        T.save(self.option_q_target.state_dict(), self.ckpt_path+"/ckpt_option_q_target_epo"+str(epo)+".pt")
        for _ in range(len(self.intra_policies)):
            T.save(self.intra_policies[_].state_dict(), self.ckpt_path+"/ckpt_intra_policy_"+str(_)+"_epo"+str(epo)+".pt")
            T.save(self.intra_critics[_].state_dict(), self.ckpt_path+"/ckpt_intra_critic_"+str(_)+"_epo"+str(epo)+".pt")

    def load_network(self, epo):
        self.option_q_target.load_state_dict(T.load(self.ckpt_path+'/ckpt_option_q_target_epo'+str(epo)+'.pt'))
        for _ in range(len(self.intra_policies)):
            self.intra_policies[_].load_state_dict(T.load(self.ckpt_path+"/ckpt_intra_policy_"+str(_)+"_epo"+str(epo)+".pt"))
            self.intra_critics[_].load_state_dict(T.load(self.ckpt_path+"/ckpt_intra_critic_"+str(_)+"_epo"+str(epo)+".pt"))

    def normalize(self, inputs):
        return self.input_rescale*((inputs - self.input_min) / (self.input_max - self.input_min))
