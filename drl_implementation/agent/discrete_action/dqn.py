import time
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.optim.rmsprop import RMSprop
from ..utils.networks_conv import DQNetwork
from ..agent_base import Agent
from ..utils.exploration_strategy import LinearDecayGreedy
from PIL import Image


# todo: use FrameStack wrapper
class DQN(Agent):
    def __init__(self, algo_params, env, transition_tuple=None, path=None, seed=-1):
        # environment
        self.env = env
        self.env.seed(seed)
        obs = self.env.reset()
        self.frame_skip = algo_params['frame_skip']
        self.original_image_shape = obs.shape
        self.image_size = algo_params['image_size']
        algo_params.update({'state_shape': (self.frame_skip, self.image_size, self.image_size),
                            'action_dim': self.env.action_space.n,
                            'init_input_means': None,
                            'init_input_vars': None
                            })
        # training args
        self.training_epoch = algo_params['training_epoch']
        self.training_frame_per_epoch = algo_params['training_frame_per_epoch']
        self.printing_gap = algo_params['printing_gap']
        self.testing_gap = algo_params['testing_gap']
        self.testing_frame_per_epoch = algo_params['testing_frame_per_epoch']
        self.saving_gap = algo_params['saving_gap']

        # args for compatibility and are NOT to be used
        algo_params['actor_learning_rate'] = 0.0
        algo_params['observation_normalization'] = False
        algo_params['tau'] = 1.0
        super(DQN, self).__init__(algo_params,
                                  transition_tuple=transition_tuple,
                                  image_obs=True,
                                  action_type='discrete',
                                  path=path,
                                  seed=seed)
        # torch
        self.network_dict.update({
            'Q': DQNetwork(self.state_shape, self.action_dim).to(self.device),
            'Q_target': DQNetwork(self.state_shape, self.action_dim).to(self.device)
        })
        self.network_keys_to_save = ['Q', 'Q_target']
        self.Q_optimizer = RMSprop(self.network_dict['Q'].parameters(),
                                   lr=self.critic_learning_rate,
                                   eps=algo_params['RMSprop_epsilon'],
                                   weight_decay=algo_params['Q_weight_decay'],
                                   centered=True)
        self._soft_update(self.network_dict['Q'], self.network_dict['Q_target'], tau=1)
        # behavioural policy args (exploration)
        epsilong_decay_frame = algo_params['epsilon_decay_fraction'] * self.training_epoch * self.training_frame_per_epoch
        self.exploration_strategy = LinearDecayGreedy(decay=epsilong_decay_frame,
                                                      rng=self.rng)
        # training args
        self.warmup_step = algo_params['warmup_step']
        self.Q_target_update_interval = algo_params['Q_target_update_interval']
        self.last_frame = None
        self.frame_buffer = [None, None, None, None]
        self.frame_count = 0
        self.reward_clip = algo_params['reward_clip']
        # statistic dict
        self.statistic_dict.update({
            'epoch_return': [],
            'epoch_test_return': []
        })

    def run(self, test=False, render=False, load_network_ep=None, sleep=0):
        if test:
            num_frames = self.testing_frame_per_epoch
            if load_network_ep is not None:
                print("Loading network parameters...")
                self._load_network(ep=load_network_ep)
            print("Start testing...")
        else:
            num_frames = self.training_frame_per_epoch
            print("Start training...")

        for epo in range(self.training_epoch):
            ep_return = self._interact(render, test, epo=epo, num_frames=num_frames, sleep=sleep)
            self.statistic_dict['epoch_return'].append(ep_return)
            print("Finished training epoch %i, " % epo, "full return %0.1f" % ep_return)

            if (epo % self.testing_gap == 0) and (epo != 0) and (not test):
                print("Evaluate agent at epoch %i..." % epo)
                ep_test_return = self._interact(render, test=True, epo=epo, num_frames=self.testing_frame_per_epoch)
                self.statistic_dict['epoch_test_return'].append(ep_test_return)
                print("Finished testing epoch %i, " % epo, "test return %0.1f" % ep_test_return)

            if (epo % self.saving_gap == 0) and (epo != 0) and (not test):
                self._save_network(ep=epo)

        if not test:
            print("Finished training")
            print("Saving statistics...")
            self._save_statistics()
            self._plot_statistics()
        else:
            print("Finished testing")

    def _interact(self, render=False, test=False, epo=0, num_frames=0, sleep=0):
        ep_return = 0
        self.frame_count = 0
        while self.frame_count < num_frames:
            done = False
            obs = self.env.reset()
            obs = self._pre_process([obs])
            num_lives = self.env.ale.lives()
            # start a new episode
            while not done:
                if render:
                    self.env.render()
                if self.env_step_count < self.warmup_step:
                    action = self.env.action_space.sample()
                else:
                    action = self._select_action(obs, test=test)

                # action repeat, aggregated reward
                frames = []
                added_reward = 0
                for _ in range(self.frame_skip):
                    new_obs, reward, done, info = self.env.step(action)
                    frames.append(new_obs.copy())
                    added_reward += reward
                time.sleep(sleep)
                # frame gray scale, resize, stack
                new_obs = self._pre_process(frames[-2:])
                # reward clipped into [-1, 1]
                reward = max(min(added_reward, self.reward_clip), -self.reward_clip)

                if num_lives > self.env.ale.lives():
                    # treat the episode as terminated when the agent loses a live in the game
                    num_lives = self.env.ale.lives()
                    done_to_save = True
                    # set the reward to be -reward_bound
                    reward = -self.reward_clip
                    # clear frame buffer when the agent starts with a new live
                    self.frame_buffer = [None, None, None, None]
                else:
                    done_to_save = done

                # return to be recorded
                ep_return += reward
                if not test:
                    self._remember(obs, action, new_obs, reward, 1 - int(done_to_save))
                    if (self.env_step_count % self.update_interval == 0) and (self.env_step_count > self.warmup_step):
                        self._learn()
                obs = new_obs
                self.frame_count += 1
                self.env_step_count += 1

                if self.frame_count % self.printing_gap == 0 and self.frame_count != 0:
                    print("Epoch %i" % epo, "passed frames %i" % self.frame_count, "return %0.1f" % ep_return)

            # clear frame buffer at the end of an episode
            self.frame_buffer = [None, None, None, None]
        return ep_return

    def _select_action(self, obs, test=False):
        if test:
            obs = T.tensor([obs], dtype=T.float32).to(self.device)
            with T.no_grad():
                action = self.network_dict['Q_target'].get_action(obs)
            return action
        else:
            if self.exploration_strategy(self.env_step_count):
                action = self.rng.integers(self.action_dim)
            else:
                obs = T.tensor([obs], dtype=T.float32).to(self.device)
                with T.no_grad():
                    action = self.network_dict['Q_target'].get_action(obs)
            return action

    def _learn(self, steps=None):
        if len(self.buffer) < self.batch_size:
            return
        if steps is None:
            steps = self.optimizer_steps

        for i in range(steps):
            if self.prioritised:
                batch, weights, inds = self.buffer.sample(self.batch_size)
                weights = T.tensor(weights).view(self.batch_size, 1).to(self.device)
            else:
                batch = self.buffer.sample(self.batch_size)
                weights = T.ones(size=(self.batch_size, 1)).to(self.device)
                inds = None

            inputs = T.tensor(batch.state, dtype=T.float32).to(self.device)
            actions = T.tensor(batch.action, dtype=T.long).unsqueeze(1).to(self.device)
            inputs_ = T.tensor(batch.next_state, dtype=T.float32).to(self.device)
            rewards = T.tensor(batch.reward, dtype=T.float32).unsqueeze(1).to(self.device)
            done = T.tensor(batch.done, dtype=T.float32).unsqueeze(1).to(self.device)

            if self.discard_time_limit:
                done = done * 0 + 1

            with T.no_grad():
                maximal_next_values = self.network_dict['Q_target'](inputs_).max(1)[0].view(self.batch_size, 1)
                value_target = rewards + done*self.gamma*maximal_next_values

            self.Q_optimizer.zero_grad()
            value_estimate = self.network_dict['Q'](inputs).gather(1, actions)
            loss = F.smooth_l1_loss(value_estimate, value_target.detach(), reduction='none')
            (loss * weights).mean().backward()
            self.Q_optimizer.step()

            if self.prioritised:
                assert inds is not None
                self.buffer.update_priority(inds, np.abs(loss.cpu().detach().numpy()))

            self.statistic_dict['critic_loss'].append(loss.detach().mean().cpu().numpy().item())

            if self.optim_step_count % self.Q_target_update_interval == 0:
                self._soft_update(self.network_dict['Q'], self.network_dict['Q_target'], tau=1)

            self.optim_step_count += 1

    def _pre_process(self, frames):
        # This method takes 2 frames and does the following things
        # 1. Max-pool two consecutive frames to deal with flickering
        # 2. Convert images to Y channel: Y = 0.299*R + 0.587*G + (1 - (0.299 + 0.587))*B
        # 3. Resize images to 84x84
        # 4. Stack it with previous frames as one observation
        # output: 1000, 1200, 1230, 1234, 2345, 3456...
        if len(frames) == 1:
            frames.insert(0, np.zeros(self.original_image_shape))
        assert len(frames) == 2

        last_img = frames[0].copy()
        img = frames[1].copy()
        img = np.max([last_img, img], axis=0)
        img = img.transpose((-1, 0, 1))
        img_Y = 0.299 * img[0] + 0.587 * img[1] + (1 - (0.299 + 0.587)) * img[2]
        img_Y_resized = np.asarray(
            Image.fromarray(img_Y).resize((self.image_size, self.image_size), Image.BILINEAR))
        for i in range(len(self.frame_buffer)):
            if self.frame_buffer[i] is None:
                self.frame_buffer[i] = img_Y_resized.copy()
                break

            if i == (len(self.frame_buffer)-1):
                del self.frame_buffer[0]
                self.frame_buffer.append(img_Y_resized.copy())

        obs = []
        for i in range(len(self.frame_buffer)):
            if self.frame_buffer[i] is not None:
                obs.append(self.frame_buffer[i].copy())
            else:
                obs.append(np.zeros((self.image_size, self.image_size)))
        return np.array(obs, dtype=np.uint8)
