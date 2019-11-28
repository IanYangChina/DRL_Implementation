# the original paper uses multiple cpu to collect data

# actor, critic networks have 3 hidden layers, each with 256 units and relu activation
# critic output without activation, while actor output with tanh and rescaling
# observation and goal are concatenated and fed into both networks
# in the original paper, actions are only x, y, z coordinates, joint velocity is set fixed

# the original paper scales observation, goals and actions into [-5, 5], and normalize to 0 mean and standard variation
# for normalization, the mean and standard deviation are computed using encountered data

# training process has 200 epochs with 50 cycles, each of which has 16 episodes and 40 optimization steps
# the total episode number is 200*50*16=160000, each of which has 50 time steps
# after every 16 episodes, 40 optimization steps are performed
# each optimization step uses a mini-batch of 128 batch size uniformly sampled from a replay buffer with 10^6 capacity
# target network is updated softly with tau=0.05
# Adam is used for learning with a learning rate of 0.001
# discount factor is 0.98, target value is clipped to [-1/(1-0.98), 0], that is [-50, 0]

# for exploration, they randomly select action from uniform distribution with 20% chance
# and with 80% chance, they add normal noise into each coordinate with standard deviation equal to 5% of the max bound

import random
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from Agent.networks import Actor, Critic
T.manual_seed(0)
random.seed(0)
np.random.seed(0)


class Normalizer(object):
    def __init__(self, input_dims, init_mean, init_var, scale_factor=5, range_max=1, range_min=-1, epsilon=1e-2, clip_range=5):
        self.input_dims = input_dims
        self.sample_count = 0
        self.history = []
        self.history_mean = init_mean
        self.history_var = init_var
        self.epsilon = epsilon*np.ones(self.input_dims)
        self.input_clip_range = (-clip_range*np.ones(self.input_dims), clip_range*np.ones(self.input_dims))
        self.scale_factor = scale_factor
        self.range_max = range_max*np.ones(self.input_dims)
        self.range_min = range_min*np.ones(self.input_dims)

    def store_history(self, *args):
        self.history.append(*args)

    # update mean and var for z-score normalization
    def update_mean(self):
        if len(self.history) == 0:
            return
        new_sample_num = len(self.history)
        new_history = np.array(self.history, dtype=np.float)
        new_mean = np.mean(new_history, axis=0)

        new_var = np.sum(np.square(new_history - new_mean), axis=0)
        new_var = (self.sample_count * self.history_var + new_var)
        new_var /= (new_sample_num + self.sample_count)
        self.history_var = np.sqrt(new_var)

        new_mean = (self.sample_count * self.history_mean + new_sample_num * new_mean)
        new_mean /= (new_sample_num + self.sample_count)
        self.history_mean = new_mean

        self.sample_count += new_sample_num
        self.history.clear()

    # pre-process inputs, currently using max-min-normalization
    def __call__(self, inputs):
        inputs = (inputs - self.history_mean) / (self.history_var+self.epsilon)
        inputs = np.clip(inputs, self.input_clip_range[0], self.input_clip_range[1])
        return self.scale_factor*inputs


class ReplayBuffer(object):
    def __init__(self, capacity, tr_namedtuple):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.episodes = []
        self.ep_position = -1
        self.Transition = tr_namedtuple

    def store_episode(self):
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

    def modify_experiences(self, k=4):
        if len(self.episodes) == 0:
            return
        for k_ in range(k):
            for _ in range(len(self.episodes)):
                ep = self.episodes[_]
                modified_ep = []
                ind = len(ep)-1-k_
                imagined_goal = ep[ind].achieved_goal
                for tr in range(ind+1):
                    s = ep[tr].state
                    dg = imagined_goal
                    a = ep[tr].action
                    ns = ep[tr].next_state
                    ag = ep[tr].achieved_goal
                    r = ep[tr].reward
                    d = ep[tr].done
                    if tr == ind:
                        modified_ep.append(self.Transition(s, dg, a, ns, ag, -0.0, 0))
                    else:
                        modified_ep.append(self.Transition(s, dg, a, ns, ag, r, d))
                self.episodes.append(modified_ep)

    def store_experience(self, new_episode, *args):
        # new_episode must be a boolean variable
        if new_episode:
            self.episodes.append([])
            self.ep_position += 1
        self.episodes[self.ep_position].append(self.Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return self.Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)


class DDPGAgent(object):
    def __init__(self, env_params, transition_namedtuple, noise_deviation_rate=0.05, random_action_chance=0.2,
                 tau=0.05, batch_size=128, memory_capacity=1000000, optimization_steps=40, clip_rate=0.98,
                 discount_factor=0.98, learning_rate=0.001):
        self.state_dim = env_params['obs_dims']
        self.goal_dim = env_params['goal_dims']
        self.action_dim = env_params['action_dims']
        self.action_max = env_params['action_max']

        self.actor = Actor(self.state_dim+self.goal_dim, self.action_dim).cuda()
        self.actor_target = Actor(self.state_dim+self.goal_dim, self.action_dim).cuda()
        self.actor_optimizer = Adam(self.actor.parameters(), lr=learning_rate)
        self.random_action_chance = random_action_chance
        self.noise_deviation = noise_deviation_rate*self.action_max
        self.normalizer = Normalizer(self.state_dim+self.goal_dim,
                                     env_params['init_input_means'], env_params['init_input_var'],
                                     scale_factor=1, range_max=1, range_min=0)

        self.critic = Critic(self.state_dim+self.goal_dim+self.action_dim).cuda()
        self.critic_target = Critic(self.state_dim+self.goal_dim+self.action_dim).cuda()
        self.critic_optimizer = Adam(self.critic.parameters(), lr=learning_rate)
        self.clip_rate = clip_rate
        self.clip_value = -1 / (1-self.clip_rate)

        self.buffer = ReplayBuffer(memory_capacity, transition_namedtuple)
        self.batch_size = batch_size

        self.optimizer_steps = optimization_steps
        self.gamma = discount_factor
        self.tau = tau
        self.soft_update(tau=1)

    def act(self, state, desired_goal, test=False):
        if not test:
            inputs = np.concatenate((state, desired_goal), axis=0)
            self.normalizer.store_history(inputs)
            chance = random.uniform(0, 1)
            if chance < self.random_action_chance:
                action = np.random.uniform(-self.action_max, self.action_max, size=(self.action_dim,))
                return action
            else:
                self.actor_target.eval()
                inputs = self.normalizer(inputs)
                inputs = T.tensor(inputs, dtype=T.float).cuda()
                action = self.actor_target(inputs).cpu().detach().numpy()
                # sigma * np.random.randn(...) + mu
                noise = self.noise_deviation*np.random.randn(self.action_dim)
                action += noise
                action = np.clip(action, -self.action_max, self.action_max)
                return action
        else:
            self.actor_target.eval()
            inputs = np.concatenate((state, desired_goal), axis=0)
            self.normalizer.store_history(inputs)
            inputs = self.normalizer(inputs)
            inputs = T.tensor(inputs, dtype=T.float).cuda()
            action = self.actor_target(inputs).cpu().detach().numpy()
            action = np.clip(action, -self.action_max, self.action_max)
            return action

    def remember(self, new_episode, *args):
        self.buffer.store_experience(new_episode, *args)

    def apply_hindsight(self):
        self.buffer.modify_experiences()
        self.buffer.store_episode()

    def learn(self, steps=None, batch_size=None):
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
            actor_inputs = T.tensor(actor_inputs, dtype=T.float32).cuda()
            actions = T.tensor(batch.action, dtype=T.float32).cuda()
            actor_inputs_ = np.concatenate((batch.next_state, batch.desired_goal), axis=1)
            actor_inputs_ = self.normalizer(actor_inputs_)
            actor_inputs_ = T.tensor(actor_inputs_, dtype=T.float32).cuda()
            rewards = T.tensor(batch.reward, dtype=T.float32).unsqueeze(1).cuda()
            done = T.tensor(batch.done, dtype=T.float32).unsqueeze(1).cuda()

            self.actor_target.eval()
            self.critic_target.eval()
            self.actor.eval()
            self.critic.train()
            self.critic_optimizer.zero_grad()
            actions_ = self.actor_target(actor_inputs_)
            critic_inputs = T.cat((actor_inputs, actions), dim=1).cuda()
            value_estimate = self.critic(critic_inputs)
            critic_inputs_ = T.cat((actor_inputs_, actions_), dim=1).cuda()
            value_ = self.critic_target(critic_inputs_)
            value_target = rewards + done*self.gamma*value_
            value_target = T.clamp(value_target, self.clip_value, 0.0)
            critic_loss = F.mse_loss(value_estimate, value_target)
            critic_loss.backward()
            self.critic_optimizer.step()

            self.critic.eval()
            self.actor.train()
            self.actor_optimizer.zero_grad()
            new_actions = self.actor(actor_inputs)
            critic_eval_inputs = T.cat((actor_inputs, new_actions), dim=1).cuda()
            actor_loss = -self.critic(critic_eval_inputs)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor.eval()

            self.soft_update()

    def soft_update(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def save_networks(self, folder, epoch):
        T.save(self.actor.state_dict(), './'+folder+'/ckpt_actor_epoch'+str(epoch)+'.pt')
        T.save(self.critic.state_dict(), './'+folder+'/ckpt_critic_epoch'+str(epoch)+'.pt')
        T.save(self.actor_target.state_dict(), './'+folder+'/ckpt_actor_target_epoch'+str(epoch)+'.pt')
        T.save(self.critic_target.state_dict(), './'+folder+'/ckpt_critic_target_epoch'+str(epoch)+'.pt')

    def load_network(self, folder, epoch):
        self.actor_target.load_state_dict(T.load('./'+folder+'/ckpt_actor_target_epoch'+str(epoch)+'.pt'))
        self.critic_target.load_state_dict(T.load('./'+folder+'/ckpt_critic_target_epoch'+str(epoch)+'.pt'))
