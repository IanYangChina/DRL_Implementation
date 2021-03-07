import time
import numpy as np
import torch as T
import torch.nn.functional as F
import kornia.augmentation as aug
from torch.optim.adam import Adam
from ..utils.env_wrapper import FrameStack, PixelPybulletGym
from ..utils.networks_conv import PixelEncoder, StochasticConvActor, ConvCritic
from ..agent_base import Agent
T.backends.cudnn.benchmark = True
# scaler = T.cuda.amp.GradScaler()


class SACDrQ(Agent):
    # https://arxiv.org/abs/2004.13649
    def __init__(self, algo_params, env, transition_tuple=None, path=None, seed=-1):
        # environment
        self.env = PixelPybulletGym(env,
                                    image_size=algo_params['image_resize_size'],
                                    crop_size=algo_params['image_crop_size'])
        self.frame_stack = algo_params['frame_stack']
        self.env = FrameStack(self.env, k=self.frame_stack)
        self.env.seed(seed)
        obs = self.env.reset()
        algo_params.update({'state_shape': obs.shape,  # make sure the shape is like (C, H, W), not (H, W, C)
                            'action_dim': self.env.action_space.shape[0],
                            'action_max': self.env.action_space.high,
                            'action_scaling': self.env.action_space.high[0],
                            })
        # training args
        self.max_env_step = algo_params['max_env_step']
        self.testing_gap = algo_params['testing_gap']
        self.testing_episodes = algo_params['testing_episodes']
        self.saving_gap = algo_params['saving_gap']

        super(SACDrQ, self).__init__(algo_params,
                                     transition_tuple=transition_tuple,
                                     image_obs=True,
                                     training_mode='step_based',
                                     path=path,
                                     seed=seed)
        # torch
        self.encoder = PixelEncoder(self.state_shape)
        self.encoder_target = PixelEncoder(self.state_shape)
        self.network_dict.update({
            'actor': StochasticConvActor(self.action_dim, encoder=self.encoder, detach_obs_encoder=True).to(self.device),
            'critic_1': ConvCritic(self.action_dim, encoder=self.encoder, detach_obs_encoder=False).to(self.device),
            'critic_1_target': ConvCritic(self.action_dim, encoder=self.encoder_target, detach_obs_encoder=True).to(self.device),
            'critic_2': ConvCritic(self.action_dim, encoder=self.encoder, detach_obs_encoder=False).to(self.device),
            'critic_2_target': ConvCritic(self.action_dim, encoder=self.encoder_target, detach_obs_encoder=True).to(self.device),
            'alpha': algo_params['alpha'],
            'log_alpha': T.tensor(np.log(algo_params['alpha']), requires_grad=True, device=self.device),
        })
        self.network_keys_to_save = ['actor']
        self.actor_optimizer = Adam(self.network_dict['actor'].parameters(), lr=self.actor_learning_rate)
        self.critic_1_optimizer = Adam(self.network_dict['critic_1'].parameters(), lr=self.critic_learning_rate)
        self.critic_2_optimizer = Adam(self.network_dict['critic_2'].parameters(), lr=self.critic_learning_rate)
        self._soft_update(self.network_dict['critic_1'], self.network_dict['critic_1_target'], tau=1)
        self._soft_update(self.network_dict['critic_2'], self.network_dict['critic_2_target'], tau=1)
        self.target_entropy = -self.action_dim
        self.alpha_optimizer = Adam([self.network_dict['log_alpha']], lr=self.actor_learning_rate)
        # augmentation args
        self.image_random_shift = T.nn.Sequential(T.nn.ReplicationPad2d(4), aug.RandomCrop(self.state_shape[-2:]))
        self.q_regularisation_k = algo_params['q_regularisation_k']
        # training args
        self.warmup_step = algo_params['warmup_step']
        self.actor_update_interval = algo_params['actor_update_interval']
        self.critic_target_update_interval = algo_params['critic_target_update_interval']
        # statistic dict
        self.statistic_dict.update({
            'episode_return': [],
            'env_step_return': [],
            'env_step_test_return': [],
            'alpha': [],
            'policy_entropy': [],
        })

    def run(self, test=False, render=False, load_network_ep=None, sleep=0):
        if test:
            num_episode = self.testing_episodes
            if load_network_ep is not None:
                print("Loading network parameters...")
                self._load_network(ep=load_network_ep)
            print("Start testing...")
            for ep in range(num_episode):
                ep_return = self._interact(render, test, sleep=sleep)
                self.statistic_dict['episode_return'].append(ep_return)
                print("Episode %i" % ep, "return %0.1f" % ep_return)
            print("Finished testing")
        else:
            print("Start training...")
            step_returns = 0
            while self.env_step_count < self.max_env_step:
                ep_return = self._interact(render, test, sleep=sleep)
                step_returns += ep_return
                if self.env_step_count % 1000 == 0:
                    # cumulative rewards every 1000 env steps
                    self.statistic_dict['env_step_return'].append(step_returns)
                    print("Env step %i" % self.env_step_count,
                          "avg return %0.1f" % self.statistic_dict['env_step_return'][-1])
                    step_returns = 0

                if (self.env_step_count % self.testing_gap == 0) and (self.env_step_count != 0) and (not test):
                    ep_test_return = []
                    for test_ep in range(self.testing_episodes):
                        ep_test_return.append(self._interact(render, test=True))
                    self.statistic_dict['env_step_test_return'].append(sum(ep_test_return) / self.testing_episodes)
                    print("Env step %i" % self.env_step_count,
                          "test return %0.1f" % (sum(ep_test_return) / self.testing_episodes))

                if (self.env_step_count % self.saving_gap == 0) and (self.env_step_count != 0) and (not test):
                    self._save_network(step=self.env_step_count)

            print("Finished training")
            print("Saving statistics...")
            self._plot_statistics(x_labels={'env_step_return': 'Environment step (x1e3)',
                                            'env_step_test_return': 'Environment step (x1e4)'},
                                  save_to_file=True)

    def _interact(self, render=False, test=False, sleep=0):
        done = False
        obs = self.env.reset()
        # build frame buffer for frame stack observations
        ep_return = 0
        # start a new episode
        while not done:
            if render:
                self.env.render()
            if self.env_step_count < self.warmup_step:
                action = self.env.action_space.sample()
            else:
                action = self._select_action(obs, test=test)
            new_obs, reward, done, info = self.env.step(action)
            time.sleep(sleep)
            ep_return += reward
            if not test:
                self._remember(obs, action, new_obs, reward, 1 - int(done))
                if (self.env_step_count % self.update_interval == 0) and (self.env_step_count > self.warmup_step):
                    self._learn()
                self.env_step_count += 1
                if self.env_step_count % 1000 == 0:
                    break
            obs = new_obs
        return ep_return

    def _select_action(self, obs, test=False):
        obs = T.as_tensor([obs], dtype=T.float, device=self.device)
        return self.network_dict['actor'].get_action(obs, mean_pi=test).detach().cpu().numpy()[0]

    def _learn(self, steps=None):
        if len(self.buffer) < self.batch_size:
            return
        if steps is None:
            steps = self.optimizer_steps

        for i in range(steps):
            if self.prioritised:
                batch, weights, inds = self.buffer.sample(self.batch_size)
                weights = T.as_tensor(weights, device=self.device).view(self.batch_size, 1)
            else:
                batch = self.buffer.sample(self.batch_size)
                weights = T.ones(size=(self.batch_size, 1), device=self.device)
                inds = None

            vanilla_actor_inputs = T.as_tensor(batch.state, dtype=T.float32, device=self.device)
            actions = T.as_tensor(batch.action, dtype=T.float32, device=self.device)
            vanilla_actor_inputs_ = T.as_tensor(batch.next_state, dtype=T.float32, device=self.device)
            rewards = T.as_tensor(batch.reward, dtype=T.float32, device=self.device).unsqueeze(1)
            done = T.as_tensor(batch.done, dtype=T.float32, device=self.device).unsqueeze(1)

            if self.discard_time_limit:
                done = done * 0 + 1

            average_value_target = 0
            for _ in range(self.q_regularisation_k):
                actor_inputs_ = self.image_random_shift(vanilla_actor_inputs_)
                with T.no_grad():
                    actions_, log_probs_ = self.network_dict['actor'].get_action(actor_inputs_, probs=True)
                    value_1_ = self.network_dict['critic_1_target'](actor_inputs_, actions_)
                    value_2_ = self.network_dict['critic_2_target'](actor_inputs_, actions_)
                    value_ = T.min(value_1_, value_2_) - (self.network_dict['alpha'] * log_probs_)
                    average_value_target = average_value_target + (rewards + done * self.gamma * value_)
            value_target = average_value_target/self.q_regularisation_k

            self.critic_1_optimizer.zero_grad()
            self.critic_2_optimizer.zero_grad()
            aggregated_critic_loss_1 = 0
            aggregated_critic_loss_2 = 0
            for _ in range(self.q_regularisation_k):
                actor_inputs = self.image_random_shift(vanilla_actor_inputs)

                value_estimate_1 = self.network_dict['critic_1'](actor_inputs, actions)
                critic_loss_1 = F.mse_loss(value_estimate_1, value_target.detach(), reduction='none')
                aggregated_critic_loss_1 = aggregated_critic_loss_1 + critic_loss_1

                value_estimate_2 = self.network_dict['critic_2'](actor_inputs, actions)
                critic_loss_2 = F.mse_loss(value_estimate_2, value_target.detach(), reduction='none')
                aggregated_critic_loss_2 = aggregated_critic_loss_2 + critic_loss_2

            # backward the both losses before calling .step(), or it will throw CudaRuntime error
            avg_critic_loss_1 = aggregated_critic_loss_1/self.q_regularisation_k
            (avg_critic_loss_1 * weights).mean().backward()
            avg_critic_loss_2 = aggregated_critic_loss_2/self.q_regularisation_k
            (avg_critic_loss_2 * weights).mean().backward()
            self.critic_1_optimizer.step()
            self.critic_2_optimizer.step()

            if self.prioritised:
                assert inds is not None
                avg_critic_loss_1 = avg_critic_loss_1.detach().cpu().numpy()
                self.buffer.update_priority(inds, np.abs(avg_critic_loss_1))

            self.statistic_dict['critic_loss'].append(avg_critic_loss_1.mean().detach())

            if self.optim_step_count % self.critic_target_update_interval == 0:
                self._soft_update(self.network_dict['critic_1'], self.network_dict['critic_1_target'])
                self._soft_update(self.network_dict['critic_2'], self.network_dict['critic_2_target'])

            if self.optim_step_count % self.actor_update_interval == 0:
                self.actor_optimizer.zero_grad()
                self.alpha_optimizer.zero_grad()
                aggregated_actor_loss = 0
                aggregated_alpha_loss = 0
                aggregated_log_probs = 0
                for _ in range(self.q_regularisation_k):
                    actor_inputs = self.image_random_shift(vanilla_actor_inputs)
                    new_actions, new_log_probs = self.network_dict['actor'].get_action(actor_inputs, probs=True)
                    aggregated_log_probs = aggregated_log_probs + new_log_probs
                    new_values = T.min(self.network_dict['critic_1'](actor_inputs, new_actions),
                                       self.network_dict['critic_2'](actor_inputs, new_actions))
                    aggregated_actor_loss = aggregated_actor_loss + (self.network_dict['alpha'] * new_log_probs - new_values).mean()
                    aggregated_alpha_loss = aggregated_alpha_loss + (self.network_dict['log_alpha'] * (-new_log_probs - self.target_entropy).detach()).mean()

                avg_actor_loss = aggregated_actor_loss/self.q_regularisation_k
                avg_actor_loss.backward()
                avg_alpha_loss = aggregated_alpha_loss/self.q_regularisation_k
                avg_alpha_loss.backward()
                self.actor_optimizer.step()
                self.alpha_optimizer.step()
                self.network_dict['alpha'] = self.network_dict['log_alpha'].exp()

                self.statistic_dict['actor_loss'].append(avg_actor_loss.mean().detach())
                self.statistic_dict['alpha'].append(self.network_dict['alpha'].detach())
                self.statistic_dict['policy_entropy'].append((-aggregated_log_probs/self.q_regularisation_k).mean().detach())

            self.optim_step_count += 1
