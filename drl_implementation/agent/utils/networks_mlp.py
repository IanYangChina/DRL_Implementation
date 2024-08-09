import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, fc1_size=256, fc2_size=256, fc3_size=256, init_w=3e-3, action_scaling=1):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.pi = nn.Linear(fc3_size, output_dim)
        self.apply(orthogonal_init)
        self.action_scaling = action_scaling

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = T.tanh(self.pi(x))
        return action * self.action_scaling


class StochasticActor(nn.Module):
    def __init__(self, input_dim, output_dim, log_std_min, log_std_max, continuous=True,
                 fc1_size=256, fc2_size=256, fc3_size=256, init_w=3e-3, action_scaling=1):
        super(StochasticActor, self).__init__()
        self.continuous = continuous
        self.action_dim = output_dim
        self.fc1 = nn.Linear(input_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        if self.continuous:
            self.fc3 = nn.Linear(fc2_size, fc3_size)
            self.mean = nn.Linear(fc3_size, output_dim)
            self.log_std = nn.Linear(fc3_size, output_dim)
        else:
            self.fc3 = nn.Linear(fc2_size, output_dim)
        self.apply(orthogonal_init)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_scaling = action_scaling

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        if self.continuous:
            mean = self.mean(x)
            log_std = self.log_std(x)
            log_std = T.clamp(log_std, self.log_std_min, self.log_std_max)
            return mean, log_std
        else:
            return x
    
    def get_action(self, inputs, std_scale=None, epsilon=1e-6, mean_pi=False, greedy=False, probs=False, entropy=False):
        if self.continuous:
            mean, log_std = self(inputs)
            if mean_pi:
                return T.tanh(mean)
            std = log_std.exp()
            if std_scale is not None:
                std *= std_scale
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
        else:
            logits = self(inputs)
            if greedy:
                actions = T.argmax(logits, dim=1, keepdim=True)
                return actions, None, None
            action_probs = F.softmax(logits, dim=1)
            dist = Categorical(action_probs)
            actions = dist.sample().view(-1, 1)
            log_probs = T.log(action_probs + epsilon).gather(1, actions)
            entropy = dist.entropy()
            return actions, log_probs, entropy

    def get_log_probs(self, inputs, actions, std_scale=None):
        actions /= self.action_scaling
        mean, log_std = self(inputs)
        std = log_std.exp()
        if std_scale is not None:
            std *= std_scale
        mu = Normal(mean, std)
        log_probs = mu.log_prob(actions)
        entropy = mu.entropy()
        return log_probs, entropy


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, fc1_size=256, fc2_size=256, fc3_size=256, init_w=3e-3, softmax=False):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.v = nn.Linear(fc3_size, output_dim)
        self.apply(orthogonal_init)
        self.softmax = softmax

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.v(x)
        if not self.softmax:
            return value
        else:
            return F.softmax(value, dim=1)

    def get_action(self, inputs):
        values = self.forward(inputs)
        return T.argmax(values).item()


def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
