import pybullet_envs
import matplotlib.pyplot as plt
from drl_implementation.agent.utils.env_wrapper import PixelPybulletGym, FrameStack

env = pybullet_envs.make("Walker2DBulletEnv-v0")
env = PixelPybulletGym(env, image_size=84, crop_size=140)
# env = FrameStack(env, k=3)

obs = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs_, reward, done, info = env.step(action)
    plt.imshow(obs_.transpose((1, 2, 0)))
    plt.pause(0.00001)
