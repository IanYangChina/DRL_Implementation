import gym
env = gym.make('Breakout-v0')
obs = env.reset()

while True:
    # env.render()
    action = env.action_space.sample()
    obs_, reward, done, info = env.step(action)
    if done:
        break