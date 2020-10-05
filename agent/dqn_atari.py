import os
import numpy as np
import torch as T
import random as R
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.optim.rmsprop import RMSprop
from agent.utils.networks import *
from agent.utils.replay_buffer import *
from agent.utils.exploration_strategy import LinearDecayGreedy
from collections import namedtuple
Tr = namedtuple('Transition', ('observation', 'action', 'next_observation', 'reward', 'done'))


class DQN(object):
    def __init__(self, env_params, path=None, seed=0,
                 lr=0.00025, mem_capacity=int(1e6), batch_size=32, tau=1.0,
                 optimization_steps=1, gamma=0.99, eps_start=1, eps_end=0.1, eps_decay=1000000):
        """Seeding"""
        T.manual_seed(seed)
        R.seed(seed)
        """Path for checkpoints"""
        if path is None:
            self.ckpt_path = "ckpts"
        else:
            self.ckpt_path = path+"/ckpts"
        if not os.path.isdir(self.ckpt_path):
            os.mkdir(self.ckpt_path)
        """Check cuda availability"""
        use_cuda = T.cuda.is_available()
        self.device = T.device("cuda" if use_cuda else "cpu")
        """Input data information"""
        self.action_num = env_params['action_num']
        """Create components of the agent"""
        self.exploration = LinearDecayGreedy(start=eps_start, end=eps_end, decay=eps_decay)
        self.agent = DQNetwork(self.action_num, num_channels=4).to(self.device)
        self.target = DQNetwork(self.action_num, num_channels=4).to(self.device)
        self.optimizer = RMSprop(self.agent.parameters(), momentum=0.95, lr=lr)
        self.memory = ReplayBuffer(mem_capacity, Tr, seed=seed)
        # copy the parameters of the main network to the target network
        self.soft_update(tau=1)
        """Specify some hyper-parameters"""
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.optimization_steps = optimization_steps

    def select_action(self, obs, ts_count=None):
        """Input pre-processing"""
        obs = T.tensor([obs], dtype=T.float).to(self.device)
        """Set the target network to evaluation mode (optional in some cases)"""
        self.target.eval()
        """Compute a forward pass through the target network"""
        values = self.target(obs)
        if ts_count is None:
            """No exploration when testing"""
            action = T.argmax(values).item()
            return action
        else:
            """Process the e-greedy exploration strategy"""
            _ = R.uniform(0, 1)
            if _ < self.exploration(ts_count):
                action = R.randint(0, self.action_num-1)
            else:
                action = T.argmax(values).item()
            return action

    def learn(self, steps=None):
        if steps is None:
            steps = self.optimization_steps
        """Don't update the network if there isn't enough data"""
        if len(self.memory) < self.batch_size:
            return
        """Update a certain number of optimization steps"""
        for s in range(steps):
            """Randomly sample a batch of transitions from the buffer, and do pre-processing"""
            batch = self.memory.sample(self.batch_size)
            inputs = T.tensor(batch.observation, dtype=T.float).to(self.device)
            inputs_ = T.tensor(batch.next_observation, dtype=T.float).to(self.device)
            actions = T.tensor(batch.action, dtype=T.long).unsqueeze(1).to(self.device)
            rewards = T.tensor(batch.reward, dtype=T.float).unsqueeze(1).to(self.device)
            done = T.tensor(batch.done, dtype=T.float).unsqueeze(1).to(self.device)
            """Set the main network to training mode, and the target network to evaluation mode"""
            self.agent.train()
            self.target.eval()
            """Pass the batch through the MAIN network to get the estimated Q values of the selected actions"""
            estimated_values = self.agent(inputs).gather(1, actions)
            """Find out the maximal Q values of the NEXT state, using the TARGET network"""
            maximal_next_values = self.target(inputs_).max(1)[0].view(self.batch_size, 1)
            """Compute the estimated target Q values of the CURRENT state"""
            target_values = rewards + done*self.gamma*maximal_next_values
            """Zero the gradients stored in the pytorch computation graph"""
            self.optimizer.zero_grad()
            """Compute the smooth L1 loss"""
            loss = F.smooth_l1_loss(estimated_values, target_values)
            """Backward the loss to get the gradients for the nodes in the network"""
            loss.backward()
            """Step the optimizer"""
            self.optimizer.step()
            """Put the main network back to evaluation mode (optional)"""
            self.agent.eval()
            """Copy some proportion of the values of the parameters of the MAIN network, to the target network"""
            self.soft_update()

    def soft_update(self, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, param in zip(self.target.parameters(), self.agent.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def save_network(self, epo):
        T.save(self.agent.state_dict(), self.ckpt_path+"/ckpt_agent_epo"+str(epo)+".pt")
        T.save(self.target.state_dict(), self.ckpt_path+"/ckpt_target_epo"+str(epo)+".pt")

    def load_network(self, epo):
        self.agent.load_state_dict(T.load(self.ckpt_path+'/ckpt_agent_epo' + str(epo)+'.pt'))
        self.target.load_state_dict(T.load(self.ckpt_path+'/ckpt_target_epo'+str(epo)+'.pt'))

    def scale(self, inputs):
        """Min-max feature scaling into [0, 1] (linear transformation)"""
        _ = (inputs - self.input_min) / (self.input_max - self.input_min)
        return _