import os
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from agent.utils.normalizer import Normalizer
from agent.utils.networks import StochasticActor, Critic
from agent.utils.replay_buffer import *
from collections import namedtuple
t = namedtuple("transition", ('state', 'action', 'log_prob', 'next_state', 'reward', 'done'))


class PPOAgent(object):
    """
    This PPO implementation follows the suggested setting given by https://arxiv.org/abs/2006.05990
    """
    def __init__(self, env_params, path=None, seed=0,
                 memory_capacity=int(1e6), optimization_steps=3, batch_size=128, discount_factor=0.98, learning_rate=0.0001,
                 clip_epsilon=0.25, value_loss_weight=0.5, entropy_penalty_weight=0.01, return_normalization=False,
                 GAE_lambda=0.9, max_ep_step=50, prioritised=False, discard_time_limit=False):
        T.manual_seed(seed)
        R.seed(seed)
        self.rng = np.random.default_rng(seed=seed)
        if path is None:
            self.ckpt_path = "ckpts"
        else:
            self.ckpt_path = path+"/ckpts"
        if not os.path.isdir(self.ckpt_path):
            os.mkdir(self.ckpt_path)
        use_cuda = T.cuda.is_available()
        self.device = T.device("cuda" if use_cuda else "cpu")

        self.state_dim = env_params['obs_dims']
        self.action_dim = env_params['action_dims']
        self.action_max = env_params['action_max']

        self.normalizer = Normalizer(self.state_dim,
                                     env_params['init_input_means'], env_params['init_input_var'])
        self.buffer = ReplayBuffer(memory_capacity, t, seed=seed)
        self.batch_size = batch_size
        self.optimization_steps = optimization_steps
        self.gamma = discount_factor

        self.actor = StochasticActor(self.state_dim, self.action_dim,
                                     log_std_min=-6, log_std_max=1).to(self.device)
        self.old_actor = StochasticActor(self.state_dim, self.action_dim,
                                         log_std_min=-6, log_std_max=1).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = Critic(self.state_dim, 1).to(self.device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=learning_rate)

        self.clip_epsilon = clip_epsilon
        self.value_loss_weight = value_loss_weight
        self.entropy_penalty_weight = entropy_penalty_weight
        self.return_normalization = return_normalization
        self.GAE_lambda = GAE_lambda
        self.max_ep_step = max_ep_step

        # redundant args for compatibility
        self.prioritised = prioritised
        self.discard_time_limit = discard_time_limit

    def select_action(self, state, log_probs=False, test=False):
        inputs = self.normalizer(state)
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        if log_probs:
            action, probs = self.old_actor.get_action(inputs, probs=True)
            return action.detach().cpu().numpy(), probs.detach().cpu().numpy()
        else:
            return self.old_actor.get_action(inputs, mean_pi=test).detach().cpu().numpy()

    def remember(self, *args):
        self.buffer.store_experience(*args)

    def learn(self, steps=None):
        if steps is None:
            steps = self.optimization_steps

        batch = self.buffer.full_memory

        states = self.normalizer(batch.state)
        states = T.tensor(states, dtype=T.float32).to(self.device)
        next_states = self.normalizer(batch.next_state)
        next_states = T.tensor(next_states, dtype=T.float32).to(self.device)
        actions = T.tensor(batch.action, dtype=T.float32).to(self.device)
        log_probs = T.tensor(batch.log_prob, dtype=T.float32).to(self.device)
        rewards = T.tensor(batch.reward, dtype=T.float32).unsqueeze(1).to(self.device)
        done = T.tensor(batch.done, dtype=T.float32).unsqueeze(1).to(self.device)

        # compute N-step returns
        returns = []
        discounted_return = 0
        rewards = rewards.flip(0)
        done = done.flip(0)
        for i in range(rewards.shape[0]):
            # done flags are stored as 0/1 integers, where 0 represents a done state
            if done[i] == 0:
                discounted_return = 0
            discounted_return = rewards[i] + self.gamma * discounted_return
            # insert n-step returns top-down
            returns.insert(0, discounted_return)
        returns = T.tensor(returns, dtype=T.float32).unsqueeze(1).to(self.device)
        if self.return_normalization:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        for step in range(steps):
            log_probs_, entropy = self.actor.get_log_probs(states, actions)
            state_values = self.critic(states)
            next_state_values = self.critic(next_states)

            ratio = T.exp(log_probs_ - log_probs.detach())
            # no done flag trick
            returns = returns + self.gamma * next_state_values.detach()
            advantages = returns - state_values.detach()

            # compute general advantage esimator
            GAE = []
            gae_t = 0
            advantages.flip(0)
            for i in range(rewards.shape[0]):
                if done[i] == 0:
                    gae_t = 0
                gae_t = advantages[i] + self.GAE_lambda * gae_t
                GAE.insert(0, (1-self.GAE_lambda)*gae_t)
            GAE = T.tensor(GAE, dtype=T.float32).unsqueeze(1).to(self.device)

            L_clip = T.min(
                ratio*GAE,
                T.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)*GAE
            )
            loss = -(L_clip -
                     self.value_loss_weight*F.mse_loss(state_values, returns))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.mean().backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.old_actor.load_state_dict(self.actor.state_dict())
        self.buffer.clear_memory()

    def save_networks(self, epoch):
        T.save(self.actor.state_dict(), self.ckpt_path+'/ckpt_actor_epoch'+str(epoch)+'.pt')
        T.save(self.critic.state_dict(), self.ckpt_path+'/ckpt_critic_epoch'+str(epoch)+'.pt')

    def load_network(self, epoch):
        self.actor.load_state_dict(T.load(self.ckpt_path+'/ckpt_actor_epoch'+str(epoch)+'.pt'))
        self.critic.load_state_dict(T.load(self.ckpt_path + '/ckpt_critic_epoch' + str(epoch) + '.pt'))
