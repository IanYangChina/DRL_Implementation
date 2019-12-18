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
    def __init__(self, input_dim, output_dim=1, fc1_size=256, fc2_size=256, fc3_size=256, init_w=3e-3):
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


class Mlp(nn.Module):
    def __init__(self, input_dim, output_dim, fc1_size=256, fc2_size=256, fc3_size=256, init_w=3e-3):
        super(Mlp, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.v = nn.Linear(fc3_size, self.output_dim)
        # initialize weight and bias of the final layer to make near-0 outputs
        self.v.weight.data.uniform_(-init_w, init_w)
        self.v.bias.data.uniform_(-init_w, init_w)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        q_values = self.v(x)
        return q_values


class OptionNet(nn.Module):
    def __init__(self, input_dim, action_dim, option_num, fc1_size=256, fc2_size=256, fc3_size=256, init_w=3e-3):
        super(OptionNet, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.option_num = option_num
        self.fc1 = nn.Linear(self.input_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
