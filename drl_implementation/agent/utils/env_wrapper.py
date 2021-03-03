import gym
import numpy as np
from skimage.transform import resize
from collections import deque


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype)
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


class PixelPybulletGym(gym.Wrapper):
    def __init__(self, env, image_size, crop_size, channel_first=True):
        gym.Wrapper.__init__(self, env)
        self.image_size = image_size
        self.crop_size = crop_size
        self.channel_first = channel_first
        self.vertical_boundary = int((env.env._render_height - self.crop_size) / 2)
        self.horizontal_boundary = int((env.env._render_width - self.crop_size) / 2)
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        self.env.reset()
        return self._get_obs()

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        # H, W, C
        obs = self.render(mode="rgb_array")
        obs = obs[self.vertical_boundary:-self.vertical_boundary, self.horizontal_boundary:-self.horizontal_boundary, :]
        obs = resize(obs, (self.image_size, self.image_size))
        if self.channel_first:
            obs = obs.transpose((-1, 0, 1))
        return obs
