import os
import random as R
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from agent.networks import Actor, Critic
from agent.replay_buffer import ReplayBuffer


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


class HindsightReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, tr_namedtuple, seed=0):
        ReplayBuffer.__init__(capacity, tr_namedtuple, seed)

    def modify_episodes(self, k=4):
        if len(self.episodes) == 0:
            return
        for k_ in range(k):
            for _ in range(len(self.episodes)):
                ep = self.episodes[_]
                if len(ep) < 2+k_:
                    continue
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


class DDPGAgent(object):
    def __init__(self, env_params, transition_namedtuple, noise_deviation_rate=0.05, random_action_chance=0.2,
                 torch_seed=0, random_seed=0,
                 tau=0.05, batch_size=128, memory_capacity=1000000, optimization_steps=40, clip_rate=0.98,
                 discount_factor=0.98, learning_rate=0.001, path=None):
        T.manual_seed(torch_seed)
        R.seed(random_seed)
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

        self.actor = Actor(self.state_dim+self.goal_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.state_dim+self.goal_dim, self.action_dim).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=learning_rate)
        self.random_action_chance = random_action_chance
        self.noise_deviation = noise_deviation_rate*self.action_max
        self.normalizer = Normalizer(self.state_dim+self.goal_dim,
                                     env_params['init_input_means'], env_params['init_input_var'],
                                     scale_factor=1, range_max=1, range_min=0)

        self.critic = Critic(self.state_dim+self.goal_dim+self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim+self.goal_dim+self.action_dim).to(self.device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=learning_rate)
        self.clip_rate = clip_rate
        self.clip_value = -1 / (1-self.clip_rate)

        self.buffer = HindsightReplayBuffer(memory_capacity, transition_namedtuple, seed=random_seed)
        self.batch_size = batch_size

        self.optimizer_steps = optimization_steps
        self.gamma = discount_factor
        self.tau = tau
        self.soft_update(tau=1)

    def act(self, state, desired_goal, test=False):
        if not test:
            inputs = np.concatenate((state, desired_goal), axis=0)
            self.normalizer.store_history(inputs)
            chance = R.uniform(0, 1)
            if chance < self.random_action_chance:
                action = np.random.uniform(-self.action_max, self.action_max, size=(self.action_dim,))
                return action
            else:
                self.actor_target.eval()
                inputs = self.normalizer(inputs)
                inputs = T.tensor(inputs, dtype=T.float).to(self.device)
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
            inputs = T.tensor(inputs, dtype=T.float).to(self.device)
            action = self.actor_target(inputs).cpu().detach().numpy()
            action = np.clip(action, -self.action_max, self.action_max)
            return action

    def remember(self, new_episode, *args):
        self.buffer.store_experience(new_episode, *args)

    def apply_hindsight(self):
        self.buffer.modify_episodes()
        self.buffer.store_episodes()

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
            actor_inputs = T.tensor(actor_inputs, dtype=T.float32).to(self.device)
            actions = T.tensor(batch.action, dtype=T.float32).to(self.device)
            actor_inputs_ = np.concatenate((batch.next_state, batch.desired_goal), axis=1)
            actor_inputs_ = self.normalizer(actor_inputs_)
            actor_inputs_ = T.tensor(actor_inputs_, dtype=T.float32).to(self.device)
            rewards = T.tensor(batch.reward, dtype=T.float32).unsqueeze(1).to(self.device)
            done = T.tensor(batch.done, dtype=T.float32).unsqueeze(1).to(self.device)

            self.actor_target.eval()
            self.critic_target.eval()
            self.actor.eval()
            self.critic.train()
            self.critic_optimizer.zero_grad()
            actions_ = self.actor_target(actor_inputs_)
            critic_inputs = T.cat((actor_inputs, actions), dim=1).to(self.device)
            value_estimate = self.critic(critic_inputs)
            critic_inputs_ = T.cat((actor_inputs_, actions_), dim=1).to(self.device)
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
            critic_eval_inputs = T.cat((actor_inputs, new_actions), dim=1).to(self.device)
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

    def save_networks(self, epoch):
        T.save(self.actor.state_dict(), self.ckpt_path+'/ckpt_actor_epoch'+str(epoch)+'.pt')
        T.save(self.critic.state_dict(), self.ckpt_path+'/ckpt_critic_epoch'+str(epoch)+'.pt')
        T.save(self.actor_target.state_dict(), self.ckpt_path+'/ckpt_actor_target_epoch'+str(epoch)+'.pt')
        T.save(self.critic_target.state_dict(), self.ckpt_path+'/ckpt_critic_target_epoch'+str(epoch)+'.pt')

    def load_network(self, epoch):
        self.actor_target.load_state_dict(T.load(self.ckpt_path+'/ckpt_actor_target_epoch'+str(epoch)+'.pt'))
        self.critic_target.load_state_dict(T.load(self.ckpt_path+'/ckpt_critic_target_epoch'+str(epoch)+'.pt'))
