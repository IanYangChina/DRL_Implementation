import os
import random as R
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from agent.utils.normalizer import Normalizer
from agent.utils.networks import StochasticActor, Critic
from agent.utils.replay_buffer import HindsightReplayBuffer


class HindsightSACAgent(object):
    def __init__(self, env_params, transition_namedtuple, path=None, seed=0, hindsight=True,
                 memory_capacity=int(1e6), optimization_steps=50, tau=0.01, batch_size=128, clip_rate=0.98,
                 discount_factor=0.98, learning_rate=0.0004):
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

        self.state_dim = env_params['obs_dims']
        self.goal_dim = env_params['goal_dims']
        self.action_dim = env_params['action_dims']
        self.action_max = env_params['action_max']

        self.normalizer = Normalizer(self.state_dim+self.goal_dim,
                                     env_params['init_input_means'], env_params['init_input_var'])
        self.hindsight = hindsight
        self.buffer = HindsightReplayBuffer(memory_capacity, transition_namedtuple, sampled_goal_num=4, seed=seed)
        self.batch_size = batch_size
        self.optimizer_steps = optimization_steps
        self.gamma = discount_factor

        self.actor = StochasticActor(self.state_dim+self.goal_dim, self.action_dim,
                                     log_std_min=-20, log_std_max=2).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = Critic(self.state_dim+self.goal_dim+self.action_dim, 1).to(self.device)
        self.critic_target = Critic(self.state_dim+self.goal_dim+self.action_dim, 1).to(self.device)
        self.tau = tau
        self.critic_target_soft_update(tau=1)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=learning_rate)
        self.clip_rate = clip_rate
        self.clip_value = -1 / (1-self.clip_rate)

    def select_action(self, state, desired_goal):
        inputs = np.concatenate((state, desired_goal), axis=0)
        inputs = self.normalizer(inputs)
        inputs = T.tensor(inputs, dtype=T.float).to(self.device)
        self.actor.eval()
        return self.actor.get_action(inputs).detach().cpu().numpy()

    def remember(self, new_episode, *args):
        self.buffer.store_experience(new_episode, *args)

    def learn(self, steps=None, batch_size=None):
        if self.hindsight:
            self.buffer.modify_episodes()
        self.buffer.store_episodes()
        if batch_size is None:
            batch_size = self.batch_size
        if len(self.buffer) < batch_size:
            return
        if steps is None:
            steps = self.optimizer_steps

        for i in range(steps):
            batch = self.buffer.sample(batch_size)
            actor_inputs = np.concatenate((batch.state, batch.desired_goal), axis=1)
            actor_inputs = self.normalizer(actor_inputs)
            actor_inputs = T.tensor(actor_inputs, dtype=T.float32).to(self.device)
            actions = T.tensor(batch.action, dtype=T.float32).to(self.device)
            actor_inputs_ = np.concatenate((batch.next_state, batch.desired_goal), axis=1)
            actor_inputs_ = self.normalizer(actor_inputs_)
            actor_inputs_ = T.tensor(actor_inputs_, dtype=T.float32).to(self.device)
            rewards = T.tensor(batch.reward, dtype=T.float32).unsqueeze(1).to(self.device)
            done = T.tensor(batch.done, dtype=T.float32).unsqueeze(1).to(self.device)

            self.critic_target.eval()
            self.actor.eval()
            self.critic.train()
            actions_, log_probs_ = self.actor.get_action(actor_inputs_, probs=True)

            critic_inputs = T.cat((actor_inputs, actions), dim=1).to(self.device)
            value_estimate = self.critic(critic_inputs)
            critic_inputs_ = T.cat((actor_inputs_, actions_), dim=1).to(self.device)
            value_ = self.critic_target(critic_inputs_)
            value_target = rewards + done*self.gamma*(value_-log_probs_)
            value_target = T.clamp(value_target, self.clip_value, 0.0)
            critic_loss = F.mse_loss(value_estimate, value_target.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            self.critic_target_soft_update()

            self.critic.eval()
            self.actor.train()
            new_actions, new_log_probs = self.actor.get_action(actor_inputs, probs=True)
            critic_eval_inputs = T.cat((actor_inputs, new_actions), dim=1).to(self.device)
            actor_loss = new_log_probs - self.critic(critic_eval_inputs)
            actor_loss = actor_loss.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor.eval()

    def critic_target_soft_update(self, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def save_networks(self, epoch):
        T.save(self.actor.state_dict(), self.ckpt_path+'/ckpt_actor_epoch'+str(epoch)+'.pt')
        T.save(self.critic.state_dict(), self.ckpt_path+'/ckpt_critic_epoch'+str(epoch)+'.pt')
        T.save(self.critic_target.state_dict(), self.ckpt_path+'/ckpt_critic_target_epoch'+str(epoch)+'.pt')

    def load_network(self, epoch):
        self.actor.load_state_dict(T.load(self.ckpt_path+'/ckpt_actor_epoch'+str(epoch)+'.pt'))
        self.critic.load_state_dict(T.load(self.ckpt_path + '/ckpt_critic_epoch' + str(epoch) + '.pt'))
        self.critic_target.load_state_dict(T.load(self.ckpt_path+'/ckpt_critic_target_epoch'+str(epoch)+'.pt'))