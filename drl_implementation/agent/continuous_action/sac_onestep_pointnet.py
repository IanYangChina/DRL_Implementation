import time
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from ..utils.networks_mlp import StochasticActor
from ..utils.networks_pointnet import CriticPointNet
from ..agent_base import Agent
from collections import namedtuple


class OneStepSAC(Agent):
    def __init__(self, algo_params, env, logging=None, transition_tuple=None, path=None, seed=-1):
        # environment
        self.env = env
        self.env.seed(seed)
        obs = self.env.reset()
        algo_params.update({'state_shape': obs['observation'].shape,
                            'goal_shape': obs['desired_goal'].shape,
                            'action_dim': self.env.action_space.shape[0],
                            'action_max': self.env.action_space.high,
                            'action_scaling': self.env.action_space.high[0],
                            })
        # training args
        self.cur_ep = 0
        self.warmup_step = algo_params['warmup_step']
        self.training_episodes = algo_params['training_episodes']
        self.testing_gap = algo_params['testing_gap']
        self.testing_episodes = algo_params['testing_episodes']
        self.saving_gap = algo_params['saving_gap']
        if transition_tuple is None:
            transition_tuple = namedtuple('transition',
                                          ['state', 'desired_goal', 'action', 'achieved_goal', 'reward'])
        super(OneStepSAC, self).__init__(algo_params, non_flat_obs=True,
                                         action_type='continuous',
                                         transition_tuple=transition_tuple,
                                         goal_conditioned=True,
                                         path=path,
                                         seed=seed,
                                         logging=logging,
                                         create_logger=True)
        # torch
        self.network_dict.update({
            'actor': StochasticActor(2048, self.action_dim,
                                     fc1_size=1024, log_std_min=-6, log_std_max=1,
                                     action_scaling=self.action_scaling).to(self.device),
            'critic_1': CriticPointNet(output_dim=1, action_dim=self.action_dim).to(self.device),
            'critic_2': CriticPointNet(output_dim=1, action_dim=self.action_dim).to(self.device),
            'critic_target': CriticPointNet(output_dim=1, action_dim=self.action_dim).to(self.device),
            'alpha': algo_params['alpha'],
            'log_alpha': T.tensor(np.log(algo_params['alpha']), requires_grad=True, device=self.device),
        })
        self.network_dict['critic_target'].eval()
        self._soft_update(self.network_dict['critic_1'], self.network_dict['critic_target'], tau=1)
        self.network_keys_to_save = ['actor', 'critic_1']
        self.actor_optimizer = Adam(self.network_dict['actor'].parameters(), lr=self.actor_learning_rate)
        self.critic_1_optimizer = Adam(self.network_dict['critic_1'].parameters(), lr=self.critic_learning_rate)
        self.critic_2_optimizer = Adam(self.network_dict['critic_2'].parameters(), lr=self.critic_learning_rate)
        self.target_entropy = -self.action_dim
        self.alpha_optimizer = Adam([self.network_dict['log_alpha']], lr=self.actor_learning_rate)
        # training args
        self.actor_update_interval = algo_params['actor_update_interval']

    def run(self, test=False, render=False, load_network_ep=None, sleep=0):
        if test:
            num_episode = self.testing_episodes
            if load_network_ep is not None:
                print("Loading network parameters...")
                self._load_network(ep=load_network_ep)
            print("Start testing...")
        else:
            num_episode = self.training_episodes
            print("Start training...")
            self.logging.info("Start training...")

        for ep in range(num_episode):
            self.cur_ep = ep
            loss_info = self._interact(render, test, sleep=sleep)
            self.logger.add_scalar(tag='Task/return', scalar_value=loss_info['emd_loss'], global_step=ep)
            self.logger.add_scalar(tag='Task/heightmap_loss', scalar_value=loss_info['height_map_loss'], global_step=ep)
            GPU_memory = self.get_gpu_memory()
            self.logger.add_scalar(tag='System/Free GPU memory', scalar_value=GPU_memory[0], global_step=ep)
            try:
                self.logger.add_scalar(tag='System/Used GPU memory', scalar_value=GPU_memory[1], global_step=ep)
            except:
                pass
            print("Episode %i" % ep, "return %0.1f" % loss_info['emd_loss'],
                  "heightmap loss %0.1f" % loss_info['height_map_loss'])
            self.logging.info("Episode %i" % ep + " return %0.1f" % loss_info['emd_loss'] +
                              " heightmap loss %0.1f" % loss_info['height_map_loss'])
            if not test and self.hindsight:
                self.buffer.hindsight()

            if (ep % self.testing_gap == 0) and (ep != 0) and (not test):
                ep_test_return = []
                ep_test_heightmap_loss = []
                for test_ep in range(self.testing_episodes):
                    loss_info = self._interact(render, test=True)
                    ep_test_return.append(loss_info['emd_loss'])
                    ep_test_heightmap_loss.append(loss_info['height_map_loss'])
                self.logger.add_scalar(tag='Task/test_return',
                                       scalar_value=(sum(ep_test_return) / self.testing_episodes), global_step=ep)
                self.logger.add_scalar(tag='Task/test_heightmap_loss',
                                       scalar_value=(sum(ep_test_heightmap_loss) / self.testing_episodes),
                                       global_step=ep)
                print("Episode %i" % ep, "test return %0.1f" % (sum(ep_test_return) / self.testing_episodes),
                      "test heightmap loss %0.1f" % (sum(ep_test_heightmap_loss) / self.testing_episodes))
                self.logging.info(
                    "Episode %i" % ep + " test return %0.1f" % (sum(ep_test_return) / self.testing_episodes) +
                    " test heightmap loss %0.1f" % (sum(ep_test_heightmap_loss) / self.testing_episodes))

            if (ep % self.saving_gap == 0) and (ep != 0) and (not test):
                self._save_network(ep=ep)

        if not test:
            print("Finished training")
            self.logging.info("Finished training")
        else:
            print("Finished testing")

    def _interact(self, render=False, test=False, sleep=0):
        # An episode has only one step
        obs = self.env.reset()
        if render:
            self.env.render()
        if self.total_env_step_count < self.warmup_step:
            action = self.env.action_space.sample()
        else:
            action = self._select_action(obs, test=test)
        obs_, reward, _, info = self.env.step(action)
        time.sleep(sleep)

        if not test:
            self._remember(obs['observation'], obs['desired_goal'], action, obs_['achieved_goal'], reward,
                           new_episode=True)
            if self.total_env_step_count % self.update_interval == 0:
                self._learn()
            self.total_env_step_count += 1

        return info

    def _select_action(self, obs, test=False):
        obs_points = T.as_tensor([obs['observation']], dtype=T.float).to(self.device)
        goal_points = T.as_tensor([obs['desired_goal']], dtype=T.float).to(self.device)
        obs_point_features = self.network_dict['critic_target'].get_features(obs_points.transpose(2, 1))
        goal_point_features = self.network_dict['critic_target'].get_features(goal_points.transpose(2, 1))
        inputs = T.cat((obs_point_features, goal_point_features), dim=1)
        action = self.network_dict['actor'].get_action(inputs, mean_pi=test).detach().cpu().numpy()
        return action[0]

    def _learn(self, steps=None):
        if len(self.buffer) < self.batch_size:
            return
        if steps is None:
            steps = self.optimizer_steps

        avg_critic_1_loss = T.zeros(1, device=self.device)
        avg_critic_2_loss = T.zeros(1, device=self.device)
        avg_actor_loss = T.zeros(1, device=self.device)
        avg_alpha = T.zeros(1, device=self.device)
        avg_policy_entropy = T.zeros(1, device=self.device)
        for i in range(steps):
            if self.prioritised:
                batch, weights, inds = self.buffer.sample(self.batch_size)
                weights = T.tensor(weights).view(self.batch_size, 1).to(self.device)
            else:
                batch = self.buffer.sample(self.batch_size)
                weights = T.ones(size=(self.batch_size, 1)).to(self.device)
                inds = None

            obs = T.as_tensor(batch.state, dtype=T.float32, device=self.device).transpose(2, 1)
            goal = T.as_tensor(batch.desired_goal, dtype=T.float32, device=self.device).transpose(2, 1)
            actions = T.as_tensor(batch.action, dtype=T.float32, device=self.device)
            rewards = T.as_tensor(batch.reward, dtype=T.float32, device=self.device).unsqueeze(1)

            self.critic_1_optimizer.zero_grad()
            value_estimate_1 = self.network_dict['critic_1'](obs, actions, goal)
            critic_loss_1 = F.mse_loss(value_estimate_1, rewards, reduction='none')
            (critic_loss_1 * weights).mean().backward()
            self.critic_1_optimizer.step()

            if self.prioritised:
                assert inds is not None
                self.buffer.update_priority(inds, np.abs(critic_loss_1.cpu().detach().numpy()))

            self.critic_2_optimizer.zero_grad()
            value_estimate_2 = self.network_dict['critic_2'](obs, actions, goal)
            critic_loss_2 = F.mse_loss(value_estimate_2, rewards, reduction='none')
            (critic_loss_2 * weights).mean().backward()
            self.critic_2_optimizer.step()

            avg_critic_1_loss += critic_loss_1.detach().mean()
            avg_critic_2_loss += critic_loss_2.detach().mean()

            if self.optim_step_count % self.actor_update_interval == 0:
                self.actor_optimizer.zero_grad()
                obs_features = self.network_dict['critic_1'].get_features(obs, detach=True)
                goal_features = self.network_dict['critic_1'].get_features(goal, detach=True)
                actor_inputs = T.cat((obs_features, goal_features), dim=1)
                new_actions, new_log_probs, new_entropy = self.network_dict['actor'].get_action(actor_inputs,
                                                                                                probs=True,
                                                                                                entropy=True)
                new_values = T.min(self.network_dict['critic_1'](obs, new_actions, goal),
                                   self.network_dict['critic_2'](obs, new_actions, goal))
                actor_loss = (self.network_dict['alpha'] * new_log_probs - new_values).mean()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.alpha_optimizer.zero_grad()
                alpha_loss = (self.network_dict['log_alpha'] * (-new_log_probs - self.target_entropy).detach()).mean()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.network_dict['alpha'] = self.network_dict['log_alpha'].exp()

                avg_actor_loss += actor_loss.detach().mean()
                avg_alpha += self.network_dict['alpha'].detach()
                avg_policy_entropy += new_entropy.detach().mean()

            self.optim_step_count += 1

        self._soft_update(self.network_dict['critic_1'], self.network_dict['critic_target'], tau=1)

        self.logger.add_scalar(tag='Critic/critic_1_loss', scalar_value=avg_critic_1_loss / steps,
                               global_step=self.cur_ep)
        self.logger.add_scalar(tag='Critic/critic_2_loss', scalar_value=avg_critic_2_loss / steps,
                               global_step=self.cur_ep)
        self.logger.add_scalar(tag='Actor/actor_loss', scalar_value=avg_actor_loss / steps, global_step=self.cur_ep)
        self.logger.add_scalar(tag='Actor/alpha', scalar_value=avg_alpha / steps, global_step=self.cur_ep)
        self.logger.add_scalar(tag='Actor/policy_entropy', scalar_value=avg_policy_entropy / steps,
                               global_step=self.cur_ep)
