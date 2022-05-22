import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


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
        if obs.max() > 1.:
            obs = obs / 255.

        x = self.features(obs)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        value = self.v(x)
        return value

    def get_action(self, obs):
        values = self.forward(obs)
        return T.argmax(values).item()


class StochasticConvActor(nn.Module):
    def __init__(self, action_dim, encoder, hidden_dim=1024, log_std_min=-10, log_std_max=2, action_scaling=1,
                 detach_obs_encoder=False,
                 goal_conditioned=False, detach_goal_encoder=True):
        super(StochasticConvActor, self).__init__()

        self.action_scaling = action_scaling
        self.encoder = encoder
        self.detach_obs_encoder = detach_obs_encoder
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        trunk_input_dim = self.encoder.feature_dim
        self.goal_conditioned = goal_conditioned
        self.detach_goal_encoder = detach_goal_encoder
        if self.goal_conditioned:
            trunk_input_dim *= 2
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim)
        )

        self.apply(orthogonal_init)

    def forward(self, obs, goal=None):
        feature = self.encoder(obs, detach=self.detach_obs_encoder)
        if self.goal_conditioned:
            assert goal is not None, "need a goal image for goal-conditioned network"
            goal_feature = self.encoder(goal, detach=self.detach_goal_encoder)
            feature = T.cat((feature, goal_feature), dim=1)

        mu, log_std = self.trunk(feature).chunk(2, dim=-1)
        log_std = T.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def get_action(self, obs, goal=None, epsilon=1e-6, mean_pi=False, probs=False, entropy=False):
        mean, log_std = self(obs, goal)
        if mean_pi:
            return T.tanh(mean)
        std = log_std.exp()
        mu = Normal(mean, std)
        z = mu.rsample()
        action = T.tanh(z)
        if not probs:
            return action * self.action_scaling
        else:
            log_probs = (mu.log_prob(z) - T.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)
            if not entropy:
                return action * self.action_scaling, log_probs
            else:
                entropy = mu.entropy()
                return action * self.action_scaling, log_probs, entropy


class ConvCritic(nn.Module):
    # Modified from https://github.com/PhilipZRH/ferm
    def __init__(self, action_dim, encoder, hidden_dim=1024, detach_obs_encoder=False,
                 goal_conditioned=False, detach_goal_encoder=True):
        super(ConvCritic, self).__init__()

        self.encoder = encoder
        self.detach_obs_encoder = detach_obs_encoder
        trunk_input_dim = self.encoder.feature_dim
        self.goal_conditioned = goal_conditioned
        self.detach_goal_encoder = detach_goal_encoder
        if self.goal_conditioned:
            trunk_input_dim *= 2
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(orthogonal_init)

    def forward(self, obs, action, goal=None):
        # detach_encoder allows to stop gradient propagation to encoder
        feature = self.encoder(obs, detach=self.detach_obs_encoder)
        if self.goal_conditioned:
            assert goal is not None, "need a goal image for goal-conditioned network"
            goal_feature = self.encoder(goal, detach=self.detach_goal_encoder)
            feature = T.cat((feature, goal_feature), dim=1)
        trunk_input = T.cat([feature, action], dim=1)
        q = self.trunk(trunk_input)
        return q


class CURL(nn.Module):
    # Modified from https://github.com/PhilipZRH/ferm
    def __init__(self, z_dim, encoder, encoder_target):
        super(CURL, self).__init__()
        self.encoder = encoder
        self.encoder_target = encoder_target
        assert z_dim == self.encoder.feature_dim == self.encoder_target.feature_dim
        self.W = nn.Parameter(T.rand(z_dim, z_dim))

    def encode(self, x, detach=False, use_target=False):
        # if exponential moving average (ema) target is enabled,
        #   then compute key values using target encoder without gradient,
        #   else compute key values with the main encoder
        # from CURL https://arxiv.org/abs/2004.04136
        if use_target:
            with T.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_score(self, z_a, z_pos):
        """
        from CURL https://arxiv.org/abs/2004.04136
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multi-class cross entropy with identity matrix for labels
        """
        Wz = T.matmul(self.W, z_pos.T)  # (z_dim,B)
        score = T.matmul(z_a, Wz)  # (B,B)
        score = score - T.max(score, 1)[0][:, None]
        return score


class PixelEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim=50, num_layers=4, num_filters=32):
        # the encoder architecture adopted by SAC-AE, DrQ and CURL
        super(PixelEncoder, self).__init__()
        assert len(obs_shape) == 3
        self.obs_shape = obs_shape[-2:]
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        x = T.randn([32] + list(obs_shape))
        out_dim = self.forward_conv(x, flatten=False).shape[-1]
        self.trunk = nn.Sequential(
            nn.Linear(num_filters * out_dim * out_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.Tanh()
        )

    def forward_conv(self, obs, flatten=True):
        if obs.max() > 1.:
            obs = obs / 255.

        conv = T.relu(self.convs[0](obs))
        for i in range(1, self.num_layers):
            conv = T.relu(self.convs[i](conv))
        if flatten:
            conv = conv.reshape(conv.size(0), -1)
        return conv

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()
        h = self.trunk(h)
        return h

    def copy_conv_weights_from(self, source):
        # only copy conv layers' weights
        for i in range(self.num_layers):
            self.convs[i].weight = source.convs[i].weight
            self.convs[i].bias = source.convs[i].bias


class PixelDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim=50, num_layers=4, num_filters=32):
        # the encoder architecture adopted by SAC-AE, DrQ and CURL
        super(PixelDecoder, self).__init__()
        assert len(obs_shape) == 3
        self.obs_shape = obs_shape[-2:]
        self.feature_dim = feature_dim
        self.num_layers = num_layers

    # todo


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
