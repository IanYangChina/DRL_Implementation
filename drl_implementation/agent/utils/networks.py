import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, fc1_size=256, fc2_size=256, fc3_size=256, init_w=3e-3, action_scaling=1):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.pi = nn.Linear(fc3_size, output_dim)
        T.nn.init.uniform_(self.pi.weight.data, -init_w, init_w)
        T.nn.init.uniform_(self.pi.bias.data, -init_w, init_w)
        self.action_scaling = action_scaling

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = T.tanh(self.pi(x))
        return action * self.action_scaling


class StochasticActor(nn.Module):
    def __init__(self, input_dim, output_dim, log_std_min, log_std_max,
                 fc1_size=256, fc2_size=256, fc3_size=256, init_w=3e-3, action_scaling=1):
        super(StochasticActor, self).__init__()
        self.action_dim = output_dim
        self.fc1 = nn.Linear(input_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.mean = nn.Linear(fc3_size, output_dim)
        T.nn.init.uniform_(self.mean.weight.data, -init_w, init_w)
        T.nn.init.uniform_(self.mean.bias.data, -init_w, init_w)
        self.log_std = nn.Linear(fc3_size, output_dim)
        T.nn.init.uniform_(self.log_std.weight.data, -init_w, init_w)
        T.nn.init.uniform_(self.log_std.bias.data, -init_w, init_w)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_scaling = action_scaling

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = T.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def get_action(self, inputs, epsilon=1e-6, mean_pi=False, probs=False, entropy=False):
        mean, log_std = self(inputs)
        if mean_pi:
            return mean
        std = log_std.exp()
        mu = Normal(mean, std)
        z = mu.rsample()
        action = T.tanh(z)
        if not probs:
            return action * self.action_scaling
        else:
            if action.shape == (self.action_dim,):
                action = action.reshape((1, self.action_dim))
            log_probs = (mu.log_prob(z) - T.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)
            if not entropy:
                return action * self.action_scaling, log_probs
            else:
                entropy = mu.entropy()
                return action * self.action_scaling, log_probs, entropy

    def get_log_probs(self, inputs, actions):
        actions /= self.action_scaling
        mean, log_std = self(inputs)
        std = log_std.exp()
        mu = Normal(mean, std)
        log_probs = mu.log_prob(actions)
        entropy = mu.entropy()
        return log_probs, entropy


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
    def __init__(self, input_dim, action_num, fc1_size=64, fc2_size=128, fc3_size=64, init_w=3e-3):
        super(IntraPolicy, self).__init__()
        self.input_dim = input_dim
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
        termination = T.sigmoid(termination)
        pi = self.pi(x)
        pi = F.softmax(pi, dim=0)
        return termination, pi


class ContinuousIntraPolicy(nn.Module):
    def __init__(self, input_dim, action_num, log_std_min, log_std_max,
                 fc1_size=64, fc2_size=128, fc3_size=64, init_w=3e-3):
        super(ContinuousIntraPolicy, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_num
        self.fc1 = nn.Linear(input_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.mean = nn.Linear(fc3_size, self.action_dim)
        T.nn.init.uniform_(self.mean.weight.data, -init_w, init_w)
        T.nn.init.uniform_(self.mean.bias.data, -init_w, init_w)
        self.log_std = nn.Linear(fc3_size, self.action_dim)
        T.nn.init.uniform_(self.log_std.weight.data, -init_w, init_w)
        T.nn.init.uniform_(self.log_std.bias.data, -init_w, init_w)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.termination = nn.Linear(fc3_size, 1)
        self.termination.weight.data.uniform_(-init_w, init_w)
        self.termination.bias.data.uniform_(-init_w, init_w)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        termination = self.termination(x)
        termination = T.sigmoid(termination)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = T.clamp(log_std, self.log_std_min, self.log_std_max)
        return termination, mean, log_std

    def get_action(self, inputs, probs=False):
        termination, mean, log_std = self(inputs)
        std = log_std.exp()
        mu = Normal(mean, std)
        z = mu.sample()
        action = T.tanh(z)
        if not probs:
            return termination, action
        else:
            log_probs = mu.log_prob(z)
            return termination, action, log_probs


class DQNetwork(nn.Module):
    def __init__(self, input_shape, action_dims, init_w=3e-3):
        super(DQNetwork, self).__init__()
        self.input_shape = input_shape
        # input_shape: tuple(c, h, w)
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        x = T.randn([32] + list(input_shape))
        self.conv_out_dim = self.features(x).view(x.size(0), -1).size(1)
        self.fc = nn.Linear(self.conv_out_dim, 512)
        self.v = nn.Linear(512, action_dims)
        T.nn.init.uniform_(self.v.weight.data, -init_w, init_w)
        T.nn.init.uniform_(self.v.bias.data, -init_w, init_w)

    def forward(self, obs):
        x = self.features(obs)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        value = self.v(x)
        return value

    def get_action(self, obs):
        values = self.forward(obs)
        return T.argmax(values).item()
