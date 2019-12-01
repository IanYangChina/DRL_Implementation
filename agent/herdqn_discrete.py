import os
import math as M
import numpy as np
import torch as T
import random as R
import torch.nn.functional as F
from torch.optim.adam import Adam
from agent.networks import Critic
from agent.replay_buffer import ReplayBuffer


class HindsightReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, tr_namedtuple, sampled_goal_num=6, seed=0):
        self.k = sampled_goal_num
        ReplayBuffer.__init__(capacity, tr_namedtuple, seed)

    def modify_episodes(self):
        if len(self.episodes) == 0:
            return
        for _ in range(len(self.episodes)):
            ep = self.episodes[_]
            imagined_goals = self.sample_achieved_goal(ep)
            for n in range(len(imagined_goals[0])):
                ind = imagined_goals[0][n]
                goal = imagined_goals[1][n]
                modified_ep = []
                for tr in range(ind+1):
                    s = ep[tr].state
                    inv = ep[tr].inventory
                    dg = goal
                    a = ep[tr].action
                    ns = ep[tr].next_state
                    ninv = ep[tr].next_inventory
                    ng = goal
                    ag = ep[tr].achieved_goal
                    r = ep[tr].reward
                    d = ep[tr].done
                    if tr == ind:
                        modified_ep.append(self.Transition(s, inv, dg, a, ns, ninv, ng, ag, 1.0, 0))
                    else:
                        modified_ep.append(self.Transition(s, inv, dg, a, ns, ninv, ng, ag, r, d))
                self.episodes.append(modified_ep)

    def sample_achieved_goal(self, ep):
        goals = [[], []]
        for k_ in range(self.k):
            done = False
            count = 0
            while not done:
                count += 1
                if count > len(ep):
                    break
                ind = R.randint(0, len(ep)-1)
                goal = ep[ind].achieved_goal
                if all(not np.allclose(goal, g) for g in goals[1]):
                    goals[1].append(goal)
                    done = True
        for g in range(len(goals[1])):
            for ind_ in range(0, len(ep)-2):
                if np.allclose(ep[ind_].achieved_goal, goals[1][g]):
                    goals[0].append(ind_)
                    break
        return goals


class HindsightDQN(object):
    def __init__(self, env_params, act_tr_namedtuple, path=None, is_act_inv=True, torch_seed=1, random_seed=1,
                 action_lr=1e-5, act_mem_capacity=int(1e6), act_batch_size=512, act_tau=0.01, clip_value=5.0,
                 optimization_steps=2, gamma=0.99, eps_start=1, eps_end=0.05, eps_decay=50000):
        T.manual_seed(torch_seed)
        R.seed(random_seed)
        if path is None:
            self.ckpt_path = "ckpts"
        else:
            self.ckpt_path = path+"/ckpts"
        if not os.path.isdir(self.ckpt_path):
            os.mkdir(self.ckpt_path)
        use_cuda = T.cuda.is_available()
        self.device = T.device("cuda" if use_cuda else "cpu")

        self.act_input_dim = env_params['act_input_dim']
        self.act_output_dim = env_params['act_output_dim']
        self.act_max = env_params['act_max']
        self.input_max = env_params['input_max']
        self.input_max_no_inv = np.delete(self.input_max, [n for n in range(4, len(self.input_max))], axis=0)
        self.input_min = env_params['input_min']
        self.input_min_no_inv = np.delete(self.input_min, [n for n in range(4, len(self.input_min))], axis=0)
        self.is_act_inv = is_act_inv

        self.action_agent = Critic(self.act_input_dim, self.act_output_dim).to(self.device)
        self.action_target = Critic(self.act_input_dim, self.act_output_dim).to(self.device)
        self.action_optimizer = Adam(self.action_agent.parameters(), lr=action_lr)
        self.action_memory = HindsightReplayBuffer(act_mem_capacity, act_tr_namedtuple, seed=random_seed)
        self.act_batch_size = act_batch_size

        self.gamma = gamma
        self.clip_value = clip_value
        self.act_soft_update(tau=1)
        self.act_tau = act_tau
        self.optimization_steps = optimization_steps
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.act_eps_threshold = 1.0

    def get_eps_threshold(self, ep):
        """
        With eps_start=1, eps_end=0.05, eps_decay=50000:
            episode     threshold
            50          0.476
            150         0.136
            200         0.088
            250         0.067
            300         0.057
            350         0.053
            400         0.051
        :param ep: number of episode
        :return: greedy threshold
        """
        self.act_eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * M.exp(-1. * ep / self.eps_decay)
        return self.act_eps_threshold

    def select_action(self, act_obs, ep=None):
        if self.is_act_inv:
            inputs = np.concatenate((act_obs['state'], act_obs['desired_goal_loc'], act_obs['inventory_vector']), axis=0)
            inputs = self.scale(inputs)
        else:
            inputs = np.concatenate((act_obs['state'], act_obs['desired_goal_loc']), axis=0)
            inputs = self.scale(inputs, inv=False)
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        self.action_target.eval()
        action_values = self.action_target(inputs)
        if ep is None:
            action = T.argmax(action_values).item()
            return action
        else:
            _ = R.uniform(0, 1)
            if _ < self.get_eps_threshold(ep):
                action = R.randint(0, self.act_max)
            else:
                action = T.argmax(action_values).item()
            return action

    def apply_hindsight(self, hindsight=False):
        if hindsight:
            self.action_memory.modify_episodes()
        self.action_memory.store_episodes()

    def remember(self, new, *args):
        self.action_memory.store_experience(new, *args)

    def act_learn(self, steps=None, batch_size=None):
        if steps is None:
            steps = self.optimization_steps
        if batch_size is None:
            batch_size = self.act_batch_size
        if len(self.action_memory) < batch_size:
            return

        for s in range(steps):
            batch = self.action_memory.sample(batch_size)
            if self.is_act_inv:
                inputs = np.concatenate((batch.state, batch.desired_goal, batch.inventory), axis=1)
                inputs_ = np.concatenate((batch.next_state, batch.next_goal, batch.next_inventory), axis=1)
                inputs = self.scale(inputs)
                inputs_ = self.scale(inputs_)
            else:
                inputs = np.concatenate((batch.state, batch.desired_goal), axis=1)
                inputs_ = np.concatenate((batch.next_state, batch.next_goal), axis=1)
                inputs = self.scale(inputs, inv=False)
                inputs_ = self.scale(inputs_, inv=False)
            inputs = T.tensor(inputs, dtype=T.float).to(self.device)
            inputs_ = T.tensor(inputs_, dtype=T.float).to(self.device)

            actions = T.tensor(batch.action, dtype=T.long).unsqueeze(1).to(self.device)
            rewards = T.tensor(batch.reward, dtype=T.float).unsqueeze(1).to(self.device)
            episode_done = T.tensor(batch.done, dtype=T.float).unsqueeze(1).to(self.device)

            self.action_agent.train()
            self.action_target.eval()
            estimated_action_values = self.action_agent(inputs).gather(1, actions)
            maximal_next_action_values = self.action_target(inputs_).max(1)[0].view(batch_size, 1)
            target_action_values = rewards + episode_done*self.gamma*maximal_next_action_values
            target_action_values = T.clamp(target_action_values, 0.0, self.clip_value)

            self.action_optimizer.zero_grad()
            loss = F.smooth_l1_loss(estimated_action_values, target_action_values)
            loss.backward()
            self.action_optimizer.step()
            self.action_agent.eval()
            self.act_soft_update()

    def act_soft_update(self, tau=None):
        if tau is None:
            tau = self.act_tau
        for target_param, param in zip(self.action_target.parameters(), self.action_agent.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def save_network(self, epo):
        T.save(self.action_agent.state_dict(), self.ckpt_path+"/ckpt_action_agent_epo"+str(epo)+".pt")
        T.save(self.action_target.state_dict(), self.ckpt_path+"/ckpt_action_target_epo"+str(epo)+".pt")

    def load_network(self, epo):
        self.action_agent.load_state_dict(T.load(self.ckpt_path+'/ckpt_action_agent_epo' + str(epo)+'.pt'))
        self.action_target.load_state_dict(T.load(self.ckpt_path+'/ckpt_action_target_epo'+str(epo)+'.pt'))

    def scale(self, inputs, inv=True):
        if inv:
            ins = (inputs - self.input_min) / (self.input_max - self.input_min)
        else:
            ins = (inputs - self.input_min_no_inv) / (self.input_max_no_inv - self.input_min_no_inv)
        return ins