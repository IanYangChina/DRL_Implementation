import torch as T
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, fc1_size=256, fc2_size=256, fc3_size=256, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.pi = nn.Linear(fc3_size, output_dim)
        T.nn.init.uniform_(self.pi.weight.data, -init_w, init_w)
        T.nn.init.uniform_(self.pi.bias.data, -init_w, init_w)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = T.tanh(self.pi(x))
        return action


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, fc1_size=256, fc2_size=256, fc3_size=256, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.v = nn.Linear(fc3_size, output_dim)
        T.nn.init.uniform_(self.v.weight.data, -init_w, init_w)
        T.nn.init.uniform_(self.v.bias.data, -init_w, init_w)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.v(x)
        return value


class IntraPolicy(nn.Module):
    def __init__(self, input_dim, option_num, action_num, fc1_size=256, fc2_size=256, fc3_size=256, init_w=3e-3):
        super(IntraPolicy, self).__init__()
        self.input_dim = input_dim+option_num
        self.action_dim = action_num
        self.fc1 = nn.Linear(self.input_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.termination = nn.Linear(fc3_size, 1)
        self.termination.weight.data.uniform_(-init_w, init_w)
        self.termination.bias.data.uniform_(-init_w, init_w)
        self.pi = nn.Linear(fc3_size, action_num)
        self.pi.weight.data.uniform_(-init_w, init_w)
        self.pi.bias.data.uniform_(-init_w, init_w)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        termination = self.termination(x)
        termination = F.sigmoid(termination)
        pi = self.pi(x)
        pi = F.softmax(pi)
        return termination, pi