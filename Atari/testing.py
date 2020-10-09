import gym
env = gym.make('Breakout-v0')
obs = env.reset()

while True:
    env.render()
    action = env.action_space.sample()
    obs_, reward, done, info = env.step(action)
    if done:
        break
# import os
# import numpy as np

# array = np.ones((1000, 8, 84, 84), dtype=np.uint8)
# np.save(os.path.dirname(os.path.realpath(__file__))+"/array", array)