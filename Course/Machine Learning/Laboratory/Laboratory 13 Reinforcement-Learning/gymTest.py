#And here's the simplest way I found to display the game environment (gym 0.26, environment version v5) while your agent is playing in Jupyter. Showing every frame is too slow, but you can skip every k frames (100 in this example) to get the idea of how it's doing:

import matplotlib.pyplot as plt
import gym
from IPython import display
%matplotlib inline

env = gym.make("ALE/SpaceInvaders-v5", obs_type='rgb', render_mode='rgb_array')

obs, info = env.reset()

for i in range(20): #20000
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if i % 100 == 0:
        plt.imshow(obs)
        display.display(plt.gcf())    
        display.clear_output(wait=True)
   
    if terminated or truncated:
        env.reset()
