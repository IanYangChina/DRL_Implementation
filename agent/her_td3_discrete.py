import os
import numpy as np
import torch as T
import random as R
import torch.nn.functional as F
from torch.optim.adam import Adam
from agent.utils.networks import Critic
from agent.utils.replay_buffer import ReplayBuffer
from agent.utils.exploration_strategy import ExpDecayGreedy


class HindsightReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, tr_namedtuple, sampled_goal_num=6, seed=0):
        self.k = sampled_goal_num
        ReplayBuffer.__init__(self, capacity, tr_namedtuple, seed)

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


class HindsightTD3(object):
    def __init__(self, env_params, tr_namedtuple, path=None, seed=0,
                 lr=1e-4, mem_capacity=int(1e6), batch_size=512, tau=0.5,
                 optimization_steps=2, gamma=0.99, eps_start=1, eps_end=0.05, eps_decay=5000):
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
        
        self.input_dim = env_params['input_dim']
        self.output_dim = env_params['output_dim']
        self.max = env_params['max']
        self.input_max = env_params['input_max']
        self.input_max_no_inv = np.delete(self.input_max, [n for n in range(4, len(self.input_max))], axis=0)
        self.input_min = env_params['input_min']
        self.input_min_no_inv = np.delete(self.input_min, [n for n in range(4, len(self.input_min))], axis=0)

        self.exploration = ExpDecayGreedy(start=eps_start, end=eps_end, decay=eps_decay)
        self.agent_1 = Critic(self.input_dim, self.output_dim).to(self.device)
        self.agent_2 = Critic(self.input_dim, self.output_dim).to(self.device)
        self.optimizer_1 = Adam(self.agent_1.parameters(), lr=lr)
        self.optimizer_2 = Adam(self.agent_2.parameters(), lr=lr)
        self.target = Critic(self.input_dim, self.output_dim).to(self.device)
        self.memory = HindsightReplayBuffer(mem_capacity, tr_namedtuple, seed=seed)
        self.batch_size = batch_size

        self.gamma = gamma
        self.soft_update(tau=1)
        self.tau = tau
        self.optimization_steps = optimization_steps
    
    def select_action(self, obs, ep=None):
        inputs = np.concatenate((obs['state'], obs['desired_goal_loc'], obs['inventory_vector']), axis=0)
        inputs = self.scale(inputs)
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        self.target.eval()
        values = self.target(inputs)
        if ep is None:
            action = T.argmax(values).item()
            return action
        else:
            _ = R.uniform(0, 1)
            if _ < self.exploration(ep):
                action = R.randint(0, self.max)
            else:
                action = T.argmax(values).item()
            return action
    
    def learn(self, steps=None, batch_size=None):
        if steps is None:
            steps = self.optimization_steps
        if batch_size is None:
            batch_size = self.batch_size
        if len(self.memory) < batch_size:
            return

        for s in range(steps):
            batch = self.memory.sample(batch_size)
            inputs = np.concatenate((batch.state, batch.desired_goal, batch.inventory), axis=1)
            inputs_ = np.concatenate((batch.next_state, batch.next_goal, batch.next_inventory), axis=1)
            inputs = self.scale(inputs)
            inputs_ = self.scale(inputs_)
            inputs = T.tensor(inputs, dtype=T.float).to(self.device)
            inputs_ = T.tensor(inputs_, dtype=T.float).to(self.device)

            actions = T.tensor(batch.action, dtype=T.long).unsqueeze(1).to(self.device)
            rewards = T.tensor(batch.reward, dtype=T.float).unsqueeze(1).to(self.device)
            episode_done = T.tensor(batch.done, dtype=T.float).unsqueeze(1).to(self.device)

            self.agent_1.train()
            maximal_next_values_1 = self.agent_1(inputs_).max(1)[0].view(batch_size, 1)
            self.agent_2.train()
            maximal_next_values_2 = self.agent_2(inputs_).max(1)[0].view(batch_size, 1)

            maximal_next_values = T.min(maximal_next_values_1, maximal_next_values_2).detach()
            target_values = rewards + episode_done*self.gamma*maximal_next_values

            estimated_values_1 = self.agent_1(inputs).gather(1, actions)
            loss_1 = F.smooth_l1_loss(estimated_values_1, target_values)
            self.optimizer_1.zero_grad()
            loss_1.backward()
            self.optimizer_1.step()
            self.agent_1.eval()

            estimated_values_2 = self.agent_2(inputs).gather(1, actions)
            loss_2 = F.smooth_l1_loss(estimated_values_2, target_values)
            self.optimizer_2.zero_grad()
            loss_2.backward()
            self.optimizer_2.step()
            self.agent_2.eval()

            self.soft_update()
    
    def apply_hindsight(self, hindsight=False):
        if hindsight:
            self.memory.modify_episodes()
        self.memory.store_episodes()

    def remember(self, new, *args):
        self.memory.store_experience(new, *args)
    
    def soft_update(self, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, param in zip(self.target.parameters(), self.agent_1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def save_network(self, epo):
        T.save(self.agent_1.state_dict(), self.ckpt_path+"/ckpt_agent_1_epo"+str(epo)+".pt")
        T.save(self.agent_2.state_dict(), self.ckpt_path+"/ckpt_agent_2_epo"+str(epo)+".pt")
        T.save(self.target.state_dict(), self.ckpt_path+"/ckpt_target_epo"+str(epo)+".pt")

    def load_network(self, epo):
        self.agent_1.load_state_dict(T.load(self.ckpt_path+'/ckpt_agent_1_epo' + str(epo)+'.pt'))
        self.agent_2.load_state_dict(T.load(self.ckpt_path+'/ckpt_agent_2_epo' + str(epo)+'.pt'))
        self.target.load_state_dict(T.load(self.ckpt_path+'/ckpt_target_epo'+str(epo)+'.pt'))

    def scale(self, inputs):
        ins = (inputs - self.input_min) / (self.input_max - self.input_min)
        return ins