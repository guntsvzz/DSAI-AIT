import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os
import gymnasium as gym
from gym.wrappers import RecordVideo
from utils import GrayScaleObservation, ResizeObservation, SkipFrame
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStack

class ActorNetwork(nn.Module):

    def __init__(self,input_size,hidden_size,action_size):
        super(ActorNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1)

        self.fc1 = nn.Linear(4096,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_size)

    def forward(self,x):

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out))
        return out

class ValueNetwork(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super(ValueNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1)

        self.fc1 = nn.Linear(4096,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)

    def forward(self,x):

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
    
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

def roll_out(actor_network,task,sample_nums,value_network,init_state):
    #task.reset()
    states = []
    actions = []
    rewards = []
    is_done = False
    final_r = 0
    state = init_state

    for j in range(sample_nums):
        states.append(np.array(state))
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        log_softmax_action = actor_network(state)
        softmax_action = torch.exp(log_softmax_action)
        action = np.random.choice(ACTION_DIM,p=softmax_action.cpu().data.numpy()[0])
        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
        next_state,reward,done,_,_ = task.step(action)
        #fix_reward = -10 if done else 1
        actions.append(one_hot_action)
        rewards.append(reward)
        final_state = next_state
        state = next_state
        if done:
            is_done = True
            state, _ = task.reset()
            break
    if not is_done:
        final_state = torch.from_numpy(np.array(final_state)).float().unsqueeze(0)
        final_r = value_network(final_state).cpu().data.numpy()

    return states,actions,rewards,final_r,state

def discount_reward(r, gamma,final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def A2C():
    # init a task generator for data fetching
    # task = gym.make("CartPole-v1")
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")
    task = FrameStack(ResizeObservation(GrayScaleObservation(SkipFrame(env, skip=4)), shape=84), num_stack=4)
    input_size = task.observation_space.shape[0]
    init_state, _ = task.reset()

    # init value network
    value_network = ValueNetwork(input_size = input_size, hidden_size = 40,output_size = 1)
    value_network_optim = torch.optim.Adam(value_network.parameters(),lr=0.01)

    # init actor network
    actor_network = ActorNetwork(input_size,40,ACTION_DIM)
    actor_network_optim = torch.optim.Adam(actor_network.parameters(),lr = 0.01)

    steps =[]
    task_episodes =[]
    test_results =[]

    for step in range(STEP):
        states, actions,rewards,final_r,current_state = roll_out(actor_network,task,SAMPLE_NUMS,value_network,init_state)
        init_state = current_state
        actions_var = torch.Tensor(actions)
        states_var = torch.Tensor(states)

        # train actor network
        actor_network_optim.zero_grad()
        log_softmax_actions = actor_network(states_var)
        vs = value_network(states_var).detach()
        # calculate qs
        qs = torch.Tensor(discount_reward(rewards,0.99,final_r))

        advantages = qs - vs
        actor_network_loss = - torch.mean(torch.sum(log_softmax_actions*actions_var,1)* advantages)
        actor_network_loss.backward()
        torch.nn.utils.clip_grad_norm(actor_network.parameters(),0.5)
        actor_network_optim.step()

        # train value network
        value_network_optim.zero_grad()
        target_values = qs.unsqueeze(1)
        values = value_network(states_var)
        criterion = nn.MSELoss()
        value_network_loss = criterion(values,target_values)
        value_network_loss.backward()
        torch.nn.utils.clip_grad_norm(value_network.parameters(),0.5)
        value_network_optim.step()

        # Testing
        if (step + 1) % 50== 0:
                result = 0
                # test_task = gym.make("CartPole-v1")
                env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")
                test_task = FrameStack(ResizeObservation(GrayScaleObservation(SkipFrame(env, skip=4)), shape=84), num_stack=4)
                for test_epi in range(10):
                    state, _ = test_task.reset()
                    for test_step in range(200):
                        # print(state)
                        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
                        softmax_action = torch.exp(actor_network(state))
                        #print(softmax_action.data)
                        action = np.argmax(softmax_action.data.numpy()[0])
                        next_state,reward,done,_,_ = test_task.step(action)
                        result += reward
                        state = next_state
                        if done:
                            break
                print("step:",step+1,"test result:",result/10.0)
                steps.append(step+1)
                test_results.append(result/10)
                
    return actor_network

# Hyper Parameters
STATE_DIM = 4
ACTION_DIM = 6
STEP = 2000
SAMPLE_NUMS = 30

actor_network = A2C()

vdo_path = 'video_rl3/'
if not os.path.exists(vdo_path):
    os.mkdir(vdo_path)

# env = RecordVideo(gym.make("CartPole-v1"), vdo_path)
env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")
env = FrameStack(ResizeObservation(GrayScaleObservation(SkipFrame(env, skip=4)), shape=84), num_stack=4)

state,_ = env.reset()
for t in range(1000):
    softmax_action = torch.exp(actor_network(torch.Tensor([state])))
    #print(softmax_action.data)
    action = np.argmax(softmax_action.data.numpy()[0])
    env.render()
    state, reward, done, _, _ = env.step(action)
    if done:
        break
env.close()