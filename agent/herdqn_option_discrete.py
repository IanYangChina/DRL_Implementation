import os
import math as M
import random as R
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from agent.networks import Critic
T.manual_seed(0)
R.seed(1)


class ActionReplayBuffer(object):
    def __init__(self, capacity, tr_namedtuple):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.episodes = []
        self.ep_position = -1
        self.Transition = tr_namedtuple

    def store_experience(self, new_option=False, *args):
        # $new_episode is a boolean value
        if new_option:
            self.episodes.append([])
            self.ep_position += 1
        self.episodes[self.ep_position].append(self.Transition(*args))

    def modify_episodes(self, k=6):
        if len(self.episodes) == 0:
            return
        for _ in range(len(self.episodes)):
            ep = self.episodes[_]
            imagined_goals = self.sample_achieved_goal(ep, k)
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

    def store_episodes(self):
        if len(self.episodes) == 0:
            return
        for ep in self.episodes:
            for n in range(len(ep)):
                if len(self.memory) < self.capacity:
                    self.memory.append(None)
                self.memory[self.position] = ep[n]
                self.position = (self.position + 1) % self.capacity
        self.episodes.clear()
        self.ep_position = -1

    def sample(self, batch_size):
        batch = R.sample(self.memory, batch_size)
        return self.Transition(*zip(*batch))

    @staticmethod
    def sample_achieved_goal(ep, k):
        goals = [[], []]

        for k_ in range(k):
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

    def __len__(self):
        return len(self.memory)


class OptionReplayBuffer(object):
    def __init__(self, capacity, tr_namedtuple):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.episodes = []
        self.ep_position = -1
        self.Transition = tr_namedtuple

    def store_experience(self, new_option=False, *args):
        # $new_episode is a boolean value
        if new_option:
            self.episodes.append([])
            self.ep_position += 1
        self.episodes[self.ep_position].append(self.Transition(*args))

    def store_episodes(self):
        if len(self.episodes) == 0:
            return
        for ep in self.episodes:
            for n in range(len(ep)):
                if len(self.memory) < self.capacity:
                    self.memory.append(None)
                self.memory[self.position] = ep[n]
                self.position = (self.position + 1) % self.capacity
        self.episodes.clear()
        self.ep_position = -1

    def sample(self, batch_size):
        batch = R.sample(self.memory, batch_size)
        return self.Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)


class OptionDQN(object):
    def __init__(self, env_params, opt_tr_namedtuple, act_tr_namedtuple, path=None, is_act_inv=True,
                 option_lr=1e-3, opt_mem_capacity=int(1e6), opt_batch_size=128, opt_tau=0.01,
                 action_lr=1e-5, act_mem_capacity=int(1e6), act_batch_size=512, act_tau=0.01, clip_value=5.0,
                 optimization_steps=2, gamma=0.99, eps_start=1, eps_end=0.05, eps_decay=50000,
                 opt_eps_decay_start=None):
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
        self.input_max = env_params['input_max']
        self.input_max_no_inv = np.delete(self.input_max, [n for n in range(4, len(self.input_max))], axis=0)
        self.input_min = env_params['input_min']
        self.input_min_no_inv = np.delete(self.input_min, [n for n in range(4, len(self.input_min))], axis=0)
        self.is_act_inv = is_act_inv

        self.option_agent = Critic(self.opt_input_dim, self.opt_output_dim).to(self.device)
        self.option_target = Critic(self.opt_input_dim, self.opt_output_dim).to(self.device)
        self.option_optimizer = Adam(self.option_agent.parameters(), lr=option_lr)
        self.option_memory = OptionReplayBuffer(opt_mem_capacity, opt_tr_namedtuple)
        self.opt_batch_size = opt_batch_size

        self.action_agent = Critic(self.act_input_dim, self.act_output_dim).to(self.device)
        self.action_target = Critic(self.act_input_dim, self.act_output_dim).to(self.device)
        self.action_optimizer = Adam(self.action_agent.parameters(), lr=action_lr)
        self.action_memory = ActionReplayBuffer(act_mem_capacity, act_tr_namedtuple)
        self.act_batch_size = act_batch_size

        self.gamma = gamma
        self.clip_value = clip_value
        self.opt_tau = opt_tau
        self.act_tau = act_tau
        self.opt_soft_update(tau=1)
        self.act_soft_update(tau=1)
        self.optimization_steps = optimization_steps
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        if opt_eps_decay_start is None:
            self.opt_eps_decay_start = 0
        else:
            self.opt_eps_decay_start = opt_eps_decay_start
        self.act_eps_threshold = 1.0
        self.opt_eps_threshold = 1.0

    def get_eps_threshold(self, ep, level):
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
        :param level: option or action
        :return: greedy threshold
        """
        if level == "action":
            self.act_eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * M.exp(-1. * ep / self.eps_decay)
            return self.act_eps_threshold
        elif level == "option":
            ep -= self.opt_eps_decay_start
            if ep < 0:
                ep = 0
            self.opt_eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * M.exp(-1. * ep / self.eps_decay)
            return self.opt_eps_threshold

    def select_option(self, opt_obs, ep=None):
        inputs = np.concatenate((opt_obs['state'], opt_obs['final_goal_loc'], opt_obs['inventory_vector']), axis=0)
        inputs = self.scale(inputs)
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        self.option_target.eval()
        option_values = self.option_target(inputs)
        if ep is None:
            option = T.argmax(option_values).item()
            return option
        elif ep == "random":
            option = R.randint(0, self.opt_max)
            return option
        else:
            _ = R.uniform(0, 1)
            if _ < self.get_eps_threshold(ep, "option"):
                option = R.randint(0, self.opt_max)
            else:
                option = T.argmax(option_values).item()
            return option

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
            if _ < self.get_eps_threshold(ep, "action"):
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

    def apply_hindsight(self, hindsight=False):
        if hindsight:
            self.action_memory.modify_episodes()
        self.action_memory.store_episodes()
        self.option_memory.store_episodes()

    def learn(self, level, steps=None, batch_size=None):
        if steps is None:
            steps = self.optimization_steps

        if level == "all":
            if batch_size is None:
                batch_size = self.opt_batch_size
            for s in range(steps):
                self.opt_learn(batch_size)
                self.opt_soft_update()
                self.act_learn(batch_size)
                self.act_soft_update()

        elif level == "option":
            if batch_size is None:
                batch_size = self.opt_batch_size
            for s in range(steps):
                self.opt_learn(batch_size)
                self.opt_soft_update()

        elif level == "action":
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
        if len(self.option_memory) < batch_size:
            return
        batch = self.option_memory.sample(batch_size)
        inputs = np.concatenate((batch.state, batch.final_goal, batch.inventory), axis=1)
        inputs = self.scale(inputs)
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        inputs_ = np.concatenate((batch.next_state, batch.next_goal, batch.next_inventory), axis=1)
        inputs_ = self.scale(inputs_)
        inputs_ = T.tensor(inputs_, dtype=T.float).to(self.device)
        options = T.tensor(batch.option, dtype=T.long).unsqueeze(1).to(self.device)
        rewards = T.tensor(batch.reward, dtype=T.float).unsqueeze(1).to(self.device)
        option_done = T.tensor(batch.option_done, dtype=T.float).unsqueeze(1).to(self.device)
        episode_done = T.tensor(batch.done, dtype=T.float).unsqueeze(1).to(self.device)

        self.option_agent.train()
        self.option_target.eval()
        estimated_option_values = self.option_agent(inputs).gather(1, options)
        unchanged_next_option_values = self.option_target(inputs_).gather(1, options)
        maximal_next_option_values = self.option_target(inputs_).max(1)[0].view(batch_size, 1)
        next_option_values = option_done*unchanged_next_option_values + (1-option_done)*maximal_next_option_values
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
        if len(self.action_memory) < batch_size:
            return
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
        T.save(self.option_agent.state_dict(), self.ckpt_path+"/ckpt_option_agent_epo"+str(epo)+".pt")
        T.save(self.option_target.state_dict(), self.ckpt_path+"/ckpt_option_target_epo"+str(epo)+".pt")
        T.save(self.action_agent.state_dict(), self.ckpt_path+"/ckpt_action_agent_epo"+str(epo)+".pt")
        T.save(self.action_target.state_dict(), self.ckpt_path+"/ckpt_action_target_epo"+str(epo)+".pt")

    def load_network(self, epo):
        self.option_agent.load_state_dict(T.load(self.ckpt_path+'/ckpt_option_agent_epo' + str(epo)+'.pt'))
        self.action_agent.load_state_dict(T.load(self.ckpt_path+'/ckpt_action_agent_epo' + str(epo)+'.pt'))
        self.option_target.load_state_dict(T.load(self.ckpt_path+'/ckpt_option_target_epo'+str(epo)+'.pt'))
        self.action_target.load_state_dict(T.load(self.ckpt_path+'/ckpt_action_target_epo'+str(epo)+'.pt'))

    def scale(self, inputs, inv=True):
        if inv:
            ins = (inputs - self.input_min) / (self.input_max - self.input_min)
        else:
            ins = (inputs - self.input_min_no_inv) / (self.input_max_no_inv - self.input_min_no_inv)
        return ins