import time
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.adam import Adam
from ..utils.networks_mlp import StochasticActor
from ..utils.networks_pointnet import CriticPointNet, CriticPointNet2
from ..agent_base import Agent
from collections import namedtuple


class GPASAC(Agent):
    def __init__(self, algo_params, env, transition_tuple=None, path=None, seed=-1):
        # environment
        self.env = env
        self.env.seed(seed)
        obs = self.env.reset()
        algo_params.update({'state_shape': obs['observation'].shape,
                            'goal_shape': obs['desired_goal'].shape,
                            'discrete_action_dim': self.env.discrete_action_space.n,
                            'continuous_action_dim': self.env.continuous_action_space.shape[0],
                            'continuous_action_max': self.env.continuous_action_space.high,
                            'continuous_action_scaling': self.env.continuous_action_space.high[0],
                            })
        # training args
        self.cur_ep = 0
        self.warmup_step = algo_params['warmup_step']
        self.training_episodes = algo_params['training_episodes']
        self.testing_gap = algo_params['testing_gap']
        self.testing_episodes = algo_params['testing_episodes']
        self.saving_gap = algo_params['saving_gap']

        self.demonstrate_skills = algo_params['demonstrate_skills']
        self.demonstrate_percentage = algo_params['demonstrate_percentage']
        assert 0 < self.demonstrate_percentage < 1, "Demonstrate percentage should be between 0 and 1"
        self.n_demonstrate_episodes = int(self.demonstrate_percentage * self.training_episodes)
        self.planned_skills = algo_params['planned_skills']
        assert not (self.demonstrate_skills and self.planned_skills), "Cannot demonstrate and planned skills at the same time"
        self.skill_plan = algo_params['skill_plan']
        self.use_planned_skills = False

        if transition_tuple is None:
            transition_tuple = namedtuple("transition",
                                          ('state', 'desired_goal', 'action',
                                           'next_state', 'achieved_goal', 'reward', 'done', 'next_skill'))
        super(GPASAC, self).__init__(algo_params,
                                     non_flat_obs=True,
                                     action_type='hybrid',
                                     transition_tuple=transition_tuple,
                                     goal_conditioned=True,
                                     path=path,
                                     seed=seed,
                                     create_logger=True)
        # torch
        self.network_dict.update({
            'discrete_actor': StochasticActor(2048, self.discrete_action_dim, continuous=False,
                                              fc1_size=1024,
                                              log_std_min=-6, log_std_max=1).to(self.device),
            'continuous_actor': StochasticActor(2048 + self.discrete_action_dim, self.continuous_action_dim,
                                                fc1_size=1024,
                                                log_std_min=-6, log_std_max=1,
                                                action_scaling=self.continuous_action_scaling).to(self.device),
            'critic_1': CriticPointNet(output_dim=1, action_dim=self.discrete_action_dim+self.continuous_action_dim).to(self.device),
            'critic_1_target': CriticPointNet(output_dim=1, action_dim=self.discrete_action_dim+self.continuous_action_dim).to(self.device),
            'critic_2': CriticPointNet(output_dim=1, action_dim=self.discrete_action_dim+self.continuous_action_dim).to(self.device),
            'critic_2_target': CriticPointNet(output_dim=1, action_dim=self.discrete_action_dim+self.continuous_action_dim).to(self.device),
            'alpha_discrete': algo_params['alpha'],
            'log_alpha_discrete': T.tensor(np.log(algo_params['alpha']), requires_grad=True, device=self.device),
            'alpha_continuous': algo_params['alpha'],
            'log_alpha_continuous': T.tensor(np.log(algo_params['alpha']), requires_grad=True, device=self.device),
        })
        self.network_dict['critic_1_target'].eval()
        self.network_dict['critic_2_target'].eval()
        self.network_keys_to_save = ['discrete_actor', 'continuous_actor', 'critic_1', 'critic_1_target']
        self.discrete_actor_optimizer = Adam(self.network_dict['discrete_actor'].parameters(),
                                             lr=self.actor_learning_rate)
        self.continuous_actor_optimizer = Adam(self.network_dict['continuous_actor'].parameters(),
                                               lr=self.actor_learning_rate)
        self.critic_1_optimizer = Adam(self.network_dict['critic_1'].parameters(), lr=self.critic_learning_rate)
        self.critic_2_optimizer = Adam(self.network_dict['critic_2'].parameters(), lr=self.critic_learning_rate)
        self._soft_update(self.network_dict['critic_1'], self.network_dict['critic_1_target'], tau=1)
        self._soft_update(self.network_dict['critic_2'], self.network_dict['critic_2_target'], tau=1)
        self.target_discrete_entropy = -self.discrete_action_dim
        self.target_continuous_entropy = -self.continuous_action_dim
        self.alpha_discrete_optimizer = Adam([self.network_dict['log_alpha_discrete']], lr=self.actor_learning_rate)
        self.alpha_continuous_optimizer = Adam([self.network_dict['log_alpha_continuous']], lr=self.actor_learning_rate)
        # training args
        # self.clip_value = algo_params['clip_value']
        self.actor_update_interval = algo_params['actor_update_interval']
        self.critic_target_update_interval = algo_params['critic_target_update_interval']

    def run(self, test=False, render=False, load_network_ep=None, sleep=0):
        if test:
            if load_network_ep is not None:
                print("Loading network parameters...")
                self._load_network(ep=load_network_ep)
            print("Start testing...")
        else:
            print("Start training...")

        for ep in range(self.training_episodes):
            if self.demonstrate_skills and (ep < self.n_demonstrate_episodes):
                self.use_planned_skills = True
            elif self.planned_skills:
                self.use_planned_skills = True
            else:
                self.use_planned_skills = False
            self.cur_ep = ep
            loss_info = self._interact(render, test, sleep=sleep)
            self.logger.add_scalar(tag='Task/return', scalar_value=loss_info['emd_loss'], global_step=self.cur_ep)
            self.logger.add_scalar(tag='Task/heightmap_loss', scalar_value=loss_info['height_map_loss'], global_step=ep)
            print("Episode %i" % ep, "return %0.1f" % loss_info['emd_loss'])
            if not test and self.hindsight:
                self.buffer.hindsight()

            if (ep % self.testing_gap == 0) and (ep != 0) and (not test):
                if self.planned_skills:
                    self.use_planned_skills = True
                else:
                    self.use_planned_skills = False
                test_return = 0
                test_heightmap_loss = 0
                for test_ep in range(self.testing_episodes):
                    loss_info = self._interact(render, test=True)
                    test_return += loss_info['emd_loss']
                    test_heightmap_loss += loss_info['height_map_loss']
                self.logger.add_scalar(tag='Task/test_return',
                                       scalar_value=(test_return / self.testing_episodes), global_step=self.cur_ep)
                self.logger.add_scalar(tag='Task/test_heightmap_loss',
                                       scalar_value=(test_heightmap_loss / self.testing_episodes), global_step=self.cur_ep)

                print("Episode %i" % ep, "test avg. return %0.1f" % (test_return / self.testing_episodes))

            if (ep % self.saving_gap == 0) and (ep != 0) and (not test):
                self._save_network(ep=ep)

        if not test:
            print("Finished training")
            print("Saving statistics...")
        else:
            print("Finished testing")

    def _interact(self, render=False, test=False, sleep=0):
        done = False
        obs = self.env.reset()
        ep_return = 0
        new_episode = True
        # start a new episode
        while not done:
            if render:
                self.env.render()
            if self.total_env_step_count < self.warmup_step:
                if self.use_planned_skills:
                    discrete_action = self.skill_plan[self.env.step_count]
                else:
                    discrete_action = self.env.discrete_action_space.sample()
                continuous_action = self.env.continuous_action_space.sample()
                action = np.concatenate([[discrete_action], continuous_action], axis=0)
            else:
                action = self._select_action(obs, test=test)
            new_obs, reward, done, info = self.env.step(action)
            time.sleep(sleep)
            ep_return += reward

            next_skill = 0
            if self.planned_skills:
                try:
                    next_skill = self.skill_plan[self.env.step_count]
                except:
                    pass

            if not test:
                self._remember(obs['observation'], obs['desired_goal'], action,
                               new_obs['observation'], new_obs['achieved_goal'], reward, 1 - int(done), next_skill,
                               new_episode=new_episode)
                self.total_env_step_count += 1
                self._learn(steps=1)

            obs = new_obs
            new_episode = False

        return info

    def _select_action(self, obs, test=False):
        obs_points = T.as_tensor([obs['observation']], dtype=T.float).to(self.device)
        goal_points = T.as_tensor([obs['desired_goal']], dtype=T.float).to(self.device)
        obs_point_features = self.network_dict['critic_1_target'].get_features(obs_points.transpose(2, 1))
        goal_point_features = self.network_dict['critic_1_target'].get_features(goal_points.transpose(2, 1))
        inputs = T.cat((obs_point_features, goal_point_features), dim=1)
        if self.use_planned_skills:
            discrete_action = T.as_tensor([self.skill_plan[self.env.step_count]], dtype=T.long).to(self.device)
        else:
            discrete_action, _, _ = self.network_dict['discrete_actor'].get_action(inputs, greedy=test)
            discrete_action.type(T.long).flatten()
        discrete_action_onehot = F.one_hot(discrete_action, self.discrete_action_dim).float()
        inputs = T.cat((inputs, discrete_action_onehot), dim=1)
        continuous_action = self.network_dict['continuous_actor'].get_action(inputs, mean_pi=test).detach().cpu().numpy()
        return np.concatenate([discrete_action.detach().cpu().numpy(), continuous_action[0]], axis=0)

    def _learn(self, steps=None):
        if len(self.buffer) < self.batch_size:
            return
        if steps is None:
            steps = self.optimizer_steps

        avg_critic_1_loss = T.zeros(1, device=self.device)
        avg_critic_2_loss = T.zeros(1, device=self.device)
        avg_discrete_actor_loss = T.zeros(1, device=self.device)
        avg_discrete_alpha = T.zeros(1, device=self.device)
        avg_discrete_policy_entropy = T.zeros(1, device=self.device)
        avg_continuous_actor_loss = T.zeros(1, device=self.device)
        avg_continuous_alpha = T.zeros(1, device=self.device)
        avg_continuous_policy_entropy = T.zeros(1, device=self.device)
        for i in range(steps):
            if self.prioritised:
                batch, weights, inds = self.buffer.sample(self.batch_size)
                weights = T.as_tensor(weights, device=self.device).view(self.batch_size, 1)
            else:
                batch = self.buffer.sample(self.batch_size)
                weights = T.ones(size=(self.batch_size, 1), device=self.device)
                inds = None

            obs = T.as_tensor(batch.state, dtype=T.float32, device=self.device).transpose(2, 1)
            obs_features = self.network_dict['critic_1_target'].get_features(obs, detach=True)
            goal = T.as_tensor(batch.desired_goal, dtype=T.float32, device=self.device).transpose(2, 1)
            goal_features = self.network_dict['critic_1_target'].get_features(goal, detach=True)
            obs_ = T.as_tensor(batch.next_state, dtype=T.float32, device=self.device).transpose(2, 1)
            obs_features_ = self.network_dict['critic_1_target'].get_features(obs_, detach=True)
            actor_inputs_ = T.cat((obs_features_, goal_features), dim=1)
            actions = T.as_tensor(batch.action, dtype=T.float32, device=self.device)
            discrete_actions = actions[:, 0].type(T.long)
            discrete_actions_onehot = F.one_hot(discrete_actions, self.discrete_action_dim).float()
            actions = T.cat((discrete_actions_onehot, actions[:, 1:]), dim=1)
            rewards = T.as_tensor(np.array(batch.reward), dtype=T.float32, device=self.device).unsqueeze(1)
            done = T.as_tensor(np.array(batch.done), dtype=T.float32, device=self.device).unsqueeze(1)

            if self.discard_time_limit:
                done = done * 0 + 1

            with T.no_grad():
                if not self.planned_skills:
                    discrete_actions_, discrete_actions_log_probs_, _ = self.network_dict['discrete_actor'].get_action(
                        actor_inputs_)
                    discrete_actions_onehot_ = F.one_hot(discrete_actions_.flatten(), self.discrete_action_dim).float()
                else:
                    discrete_actions_planned_ = T.as_tensor(batch.next_skill, dtype=T.long, device=self.device)
                    discrete_actions_planned_onehot_ = F.one_hot(discrete_actions_planned_, self.discrete_action_dim).float()
                    discrete_actions_onehot_ = discrete_actions_planned_onehot_
                    discrete_actions_log_probs_ = T.ones(size=(self.batch_size, 1), device=self.device, dtype=T.float32)

                actor_inputs_ = T.cat((actor_inputs_, discrete_actions_onehot_), dim=1)
                continuous_actions_, continuous_actions_log_probs_ = self.network_dict[
                    'continuous_actor'].get_action(actor_inputs_, probs=True)
                actions_ = T.cat((discrete_actions_onehot_, continuous_actions_), dim=1)

                value_1_ = self.network_dict['critic_1_target'](obs_, actions_, goal)
                value_2_ = self.network_dict['critic_2_target'](obs_, actions_, goal)
                value_ = T.min(value_1_, value_2_) - \
                         (self.network_dict['alpha_discrete'] * discrete_actions_log_probs_) - \
                         (self.network_dict['alpha_continuous'] * continuous_actions_log_probs_)
                value_target = rewards + done * self.gamma * value_
                # value_target = T.clamp(value_target, -self.clip_value, 0.0)

            self.critic_1_optimizer.zero_grad()
            value_estimate_1 = self.network_dict['critic_1'](obs, actions, goal)
            critic_loss_1 = F.mse_loss(value_estimate_1, value_target.detach(), reduction='none')
            (critic_loss_1 * weights).mean().backward()
            self.critic_1_optimizer.step()

            if self.prioritised:
                assert inds is not None
                self.buffer.update_priority(inds, np.abs(critic_loss_1.cpu().detach().numpy()))

            self.critic_2_optimizer.zero_grad()
            value_estimate_2 = self.network_dict['critic_2'](obs, actions, goal)
            critic_loss_2 = F.mse_loss(value_estimate_2, value_target.detach(), reduction='none')
            (critic_loss_2 * weights).mean().backward()
            self.critic_2_optimizer.step()

            avg_critic_1_loss += critic_loss_1.detach().mean()
            avg_critic_2_loss += critic_loss_2.detach().mean()

            if self.optim_step_count % self.critic_target_update_interval == 0:
                self._soft_update(self.network_dict['critic_1'], self.network_dict['critic_1_target'])
                self._soft_update(self.network_dict['critic_2'], self.network_dict['critic_2_target'])

            if self.optim_step_count % self.actor_update_interval == 0:
                self.discrete_actor_optimizer.zero_grad()
                self.continuous_actor_optimizer.zero_grad()
                actor_inputs = T.cat((obs_features, goal_features), dim=1)
                if not self.planned_skills:
                    new_discrete_actions, new_discrete_action_log_probs, new_discrete_action_entropy = \
                        self.network_dict['discrete_actor'].get_action(actor_inputs)
                    new_discrete_actions_onehot = F.one_hot(new_discrete_actions.flatten(), self.discrete_action_dim).float()
                else:
                    new_discrete_actions_onehot = discrete_actions_onehot

                new_continuous_actions, new_continuous_action_log_probs, new_continuous_action_entropy = \
                    self.network_dict['continuous_actor'].get_action(
                        T.cat((actor_inputs, new_discrete_actions_onehot), dim=1), probs=True, entropy=True)
                new_actions = T.cat((new_discrete_actions_onehot, new_continuous_actions), dim=1)

                new_values = T.min(self.network_dict['critic_1'](obs, new_actions, goal),
                                   self.network_dict['critic_2'](obs, new_actions, goal))

                if not self.planned_skills:
                    discrete_actor_loss = (
                                self.network_dict['alpha_discrete'] * new_discrete_action_log_probs - new_values).mean()
                    discrete_actor_loss.backward(retain_graph=True)
                    self.discrete_actor_optimizer.step()

                    self.alpha_discrete_optimizer.zero_grad()
                    discrete_alpha_loss = (self.network_dict['log_alpha_discrete'] * (
                                -new_discrete_action_log_probs - self.target_discrete_entropy).detach()).mean()
                    discrete_alpha_loss.backward()
                    self.alpha_discrete_optimizer.step()
                    self.network_dict['alpha_discrete'] = self.network_dict['log_alpha_discrete'].exp()

                    avg_discrete_actor_loss += discrete_actor_loss.detach()
                    avg_discrete_alpha += self.network_dict['alpha_discrete'].detach()
                    avg_discrete_policy_entropy += new_discrete_action_entropy.detach().mean()

                continuous_actor_loss = (
                            self.network_dict['alpha_continuous'] * new_continuous_action_log_probs - new_values).mean()
                continuous_actor_loss.backward()
                self.continuous_actor_optimizer.step()

                self.alpha_continuous_optimizer.zero_grad()
                continuous_alpha_loss = (self.network_dict['log_alpha_continuous'] * (
                            -new_continuous_action_log_probs - self.target_continuous_entropy).detach()).mean()
                continuous_alpha_loss.backward()
                self.alpha_continuous_optimizer.step()
                self.network_dict['alpha_continuous'] = self.network_dict['log_alpha_continuous'].exp()

                avg_continuous_actor_loss += continuous_actor_loss.detach()
                avg_continuous_alpha += self.network_dict['alpha_continuous'].detach()
                avg_continuous_policy_entropy += new_continuous_action_entropy.detach().mean()

            self.optim_step_count += 1

        self.logger.add_scalar(tag='Critic/critic_1_loss', scalar_value=avg_critic_1_loss / steps, global_step=self.cur_ep)
        self.logger.add_scalar(tag='Critic/critic_2_loss', scalar_value=avg_critic_2_loss / steps, global_step=self.cur_ep)
        if not self.planned_skills:
            self.logger.add_scalar(tag='Actor/discrete_actor_loss', scalar_value=avg_discrete_actor_loss / steps, global_step=self.cur_ep)
            self.logger.add_scalar(tag='Actor/discrete_alpha', scalar_value=avg_discrete_alpha / steps, global_step=self.cur_ep)
            self.logger.add_scalar(tag='Actor/discrete_policy_entropy', scalar_value=avg_discrete_policy_entropy / steps,
                                   global_step=self.cur_ep)
        self.logger.add_scalar(tag='Actor/continuous_actor_loss', scalar_value=avg_continuous_actor_loss / steps, global_step=self.cur_ep)
        self.logger.add_scalar(tag='Actor/continuous_alpha', scalar_value=avg_continuous_alpha / steps, global_step=self.cur_ep)
        self.logger.add_scalar(tag='Actor/continuous_policy_entropy', scalar_value=avg_continuous_policy_entropy / steps,
                               global_step=self.cur_ep)
