import os
import random as R
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from agent.utils.normalizer import Normalizer
from agent.utils.replay_buffer import HindsightReplayBuffer
from agent.utils.networks import Critic, ContinuousIntraPolicy
from agent.utils.exploration_strategy import ExpDecayGreedy


class OptionCritic(object):
    def __init__(self, env_params, act_tr_namedtuple, opt_tr_namedtuple, path=None, seed=0, double_q=False,
                 option_lr=1e-5, opt_mem_capacity=int(1e6), opt_batch_size=128, opt_tau=0.5, opt_optimization_steps=10,
                 eps_start=1, eps_end=0.05, eps_decay=50000,
                 action_lr=1e-5, act_mem_capacity=int(1e6), act_batch_size=64, act_tau=0.5, act_optimization_steps=10,
                 action_entropy_beta=0.01,
                 gamma=0.98, clip_value=-50.0):
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

        self.option_num = env_params['option_num']
        self.state_dim = env_params['obs_dims']
        self.goal_dim = env_params['goal_dims']
        self.action_dim = env_params['action_dims']
        self.normalizer = Normalizer(self.state_dim + self.goal_dim,
                                     env_params['init_input_means'], env_params['init_input_vars'],
                                     scale_factor=1)

        """Goal-conditioned Intra-policies"""
        self.intra_policies = []
        self.intra_policy_optimizers = []
        self.intra_critics = []
        self.intra_critic_optimizers = []
        self.act_memories = []
        self.act_batch_size = act_batch_size
        self.act_tau = act_tau
        self.act_optimization_steps = act_optimization_steps
        for n in range(self.option_num):
            # [pi(s,o), beta(s)] * num_o
            self.intra_policies.append(ContinuousIntraPolicy(self.state_dim+self.goal_dim, self.action_dim,
                                                             log_std_min=-20, log_std_max=2).to(self.device))
            self.intra_policy_optimizers.append(Adam(self.intra_policies[n].parameters(), lr=action_lr))
            # Q(s,o) * num_o (Q_U)
            self.intra_critics.append(Critic(self.state_dim+self.goal_dim+self.action_dim).to(self.device))
            self.intra_critic_optimizers.append(Adam(self.intra_critics[n].parameters(), lr=action_lr))
            self.act_memories.append(HindsightReplayBuffer(act_mem_capacity, act_tr_namedtuple, seed=seed))
        self.action_entropy_beta = action_entropy_beta
        self.intra_policy_loss = []
        self.termination_loss = []

        """Policy over options"""
        # Q(s) over options (Q_omega)
        self.option_q = Critic(self.state_dim+self.goal_dim, self.option_num).to(self.device)
        self.option_q_target = Critic(self.state_dim+self.goal_dim, self.option_num).to(self.device)
        self.opt_exploration = ExpDecayGreedy(eps_start, eps_end, eps_decay)
        self.option_optimizer = Adam(self.option_q.parameters(), lr=option_lr)
        self.opt_optimization_steps = opt_optimization_steps
        self.option_policy_loss = []
        self.option_memory = HindsightReplayBuffer(opt_mem_capacity, opt_tr_namedtuple, seed=seed)
        self.opt_batch_size = opt_batch_size
        self.opt_tau = opt_tau
        self.opt_soft_update(tau=1)
        self.clip_value = clip_value
        self.gamma = gamma

    def select_option(self, state, desired_goal, ep=None):
        inputs = np.concatenate((state, desired_goal), axis=0)
        inputs = self.normalizer(inputs)
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

    def select_action(self, option, state, desired_goal):
        inputs = np.concatenate((state, desired_goal), axis=0)
        inputs = self.normalizer(inputs)
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        self.intra_policies[option].eval()
        termination_probs, action = self.intra_policies[option].get_action(inputs)
        termination_probs = termination_probs.detach().cpu().numpy()
        termination = np.random.choice([True, False], 1, p=[termination_probs[0], 1 - termination_probs[0]])
        action = action.detach().cpu().numpy()
        return bool(termination), action

    def intra_policy_learn(self, steps=None, batch_size=None, hindsight=False):
        action_losses = []
        termination_losses = []
        for option in range(self.option_num):
            buffer = self.act_memories[option]
            if hindsight:
                buffer.modify_experiences()
            buffer.store_episode()
            if batch_size is None:
                batch_size = self.act_batch_size
            if len(buffer) < batch_size:
                return
            if steps is None:
                steps = self.act_optimization_steps
            for i in range(steps):
                batch = buffer.sample(batch_size)
                actor_inputs = np.concatenate((batch.state, batch.desired_goal), axis=1)
                actor_inputs = self.normalizer(actor_inputs)
                actor_inputs = T.tensor(actor_inputs, dtype=T.float32).to(self.device)
                actions = T.tensor(batch.action, dtype=T.float32).to(self.device)
                actor_inputs_ = np.concatenate((batch.next_state, batch.desired_goal), axis=1)
                actor_inputs_ = self.normalizer(actor_inputs_)
                actor_inputs_ = T.tensor(actor_inputs_, dtype=T.float32).to(self.device)
                rewards = T.tensor(batch.reward, dtype=T.float32).unsqueeze(1).to(self.device)
                done = T.tensor(batch.done, dtype=T.float32).unsqueeze(1).to(self.device)

                # Update intra_policy_critic
                self.option_q_target.eval()
                self.intra_policies[option].eval()
                self.intra_critics[option].train()
                termination_, _ = self.intra_policies[option](actor_inputs_)
                estimated_U_ = (1-termination_)*self.option_q_target(actor_inputs_)[option] + termination_*self.option_q_target(actor_inputs_).max()
                target_Q_U_swa = rewards + (1-done)*self.gamma*estimated_U_
                target_Q_U_swa = T.clamp(target_Q_U_swa, self.clip_value, 0.0)
                self.intra_critic_optimizers[option].zero_grad()
                critic_inputs = T.cat((actor_inputs, actions), dim=1).to(self.device)
                estimated_Q_U_swa = self.intra_critics[option](critic_inputs)
                intra_critic_loss = F.smooth_l1_loss(estimated_Q_U_swa, target_Q_U_swa.detach())
                intra_critic_loss.backward()
                self.intra_critic_optimizers[option].step()
                self.intra_critics[option].eval()
                action_losses.append(intra_critic_loss.detach().cpu().numpy())

                # Update intra_policy
                self.intra_policies[option].train()
                termination, actions_, log_probs_ = self.intra_policies[option](actor_inputs_, probs=True)

                entropy = -T.sum(T.exp(log_probs_)*log_probs_)
                # Subtracting baseline Q_U(s, o, a) - Q_Omega(s, o)
                critic_inputs_ = T.cat((actor_inputs_, actions_), dim=1).to(self.device)
                Q_U_swa = self.intra_critics[option](critic_inputs_)
                Q_U_swa = Q_U_swa - self.option_q_target(actor_inputs_)[option]
                self.intra_policy_optimizers[option].zero_grad()
                intra_policy_loss = -log_probs_*Q_U_swa - self.action_entropy_beta*entropy
                intra_policy_loss.backward(create_graph=True)
                Q_Omega = self.option_q_target(actor_inputs_)
                A_so = Q_Omega[option] - Q_Omega.max()
                termination_loss = termination*A_so
                termination_loss.backward()
                self.intra_policy_optimizers[option].step()
                self.intra_policies[option].eval()
                termination_losses.append(termination_loss.detach().cpu().numpy())
        self.intra_policy_loss.append(np.mean(action_losses))
        self.termination_loss.append(np.mean(termination_losses))

    def opt_learn(self, batch_size=None, steps=None, hindsight=False):
        if hindsight:
            self.option_memory.modify_experiences()
        self.option_memory.store_episode()
        if batch_size is None:
            batch_size = self.opt_batch_size
        if len(self.option_memory) < (2*batch_size):
            return
        if steps is None:
            steps = self.opt_optimization_steps
        option_policy_loss_tmp = []
        for _ in range(steps):
            batch = self.option_memory.sample(batch_size)

            inputs = np.concatenate((batch.state, batch.desired_goal, batch.inventory), axis=1)
            inputs_ = np.concatenate((batch.next_state, batch.next_goal, batch.next_inventory), axis=1)
            inputs = self.normalizer(inputs)
            inputs_ = self.normalizer(inputs_)
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
            target_values = T.clamp(target_values, self.clip_value, 0.0)

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

    def save_networks(self, epo):
        T.save(self.option_q_target.state_dict(), self.ckpt_path+"/ckpt_option_q_target_epo"+str(epo)+".pt")
        for _ in range(len(self.intra_policies)):
            T.save(self.intra_policies[_].state_dict(), self.ckpt_path+"/ckpt_intra_policy_"+str(_)+"_epo"+str(epo)+".pt")
            T.save(self.intra_critics[_].state_dict(), self.ckpt_path+"/ckpt_intra_critic_"+str(_)+"_epo"+str(epo)+".pt")

    def load_networks(self, epo):
        self.option_q_target.load_state_dict(T.load(self.ckpt_path+'/ckpt_option_q_target_epo'+str(epo)+'.pt'))
        for _ in range(len(self.intra_policies)):
            self.intra_policies[_].load_state_dict(T.load(self.ckpt_path+"/ckpt_intra_policy_"+str(_)+"_epo"+str(epo)+".pt"))
            self.intra_critics[_].load_state_dict(T.load(self.ckpt_path+"/ckpt_intra_critic_"+str(_)+"_epo"+str(epo)+".pt"))