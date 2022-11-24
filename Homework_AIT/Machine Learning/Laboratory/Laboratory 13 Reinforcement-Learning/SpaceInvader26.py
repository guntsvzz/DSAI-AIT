import matplotlib.pyplot as plt
import gym
from IPython import display
%matplotlib inline

# env = gym.make("CartPole-v0")
# env = gym.make("DoubleDunk-v0")
# env = gym.make("SpaceInvaders-v0")
# env = gym.make("Acrobot-v1") # double invert pendulum
env = gym.make("ALE/SpaceInvaders-v5", obs_type='rgb', render_mode='rgb_array')

obs, info = env.reset()

for i in range(20000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if i % 100 == 0:
        plt.imshow(obs)
        display.display(plt.gcf())    
        display.clear_output(wait=True)
   
    if terminated or truncated:
        env.reset()

env.close()