# How to run latest gym, Space Invaders locally:

# To install the Atari ROMs, you should be able to do
# pip3 install cmake
# pip3 install ale-py
# pip3 install autorom
# AutoROM
# Replace the path below with the path indicated by AutoROM
# ale-import-roms ~/.local/lib/python3.10/site-packages/AutoROM/roms

# pip3 install gym[all]

# Then, to run the Space Invaders simulation: you can run the following program:

import gym

# I'm not sure if you have to do this once or not:

# import ale_py
# from ale_py import ALEInterface
# ale = ALEInterface()
# from ale_py.roms import SpaceInvaders
# ale.loadROM(SpaceInvaders)

#Test it
env = gym.make("ALE/SpaceInvaders-v5", obs_type='rgb', render_mode='human')

obs = env.reset()

for i in range(200000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        env.reset()

env.close()