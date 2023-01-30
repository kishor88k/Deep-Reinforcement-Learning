#######################################################################################################################
#    Full Gradient DQN - CartPole                                                                                     #
#    @ Mar, 2021; Reinforcement Learning                                                                              #
#    Deep learning Framework                                                                                          #
#    Train the DQN Agent                                                                                              #
#######################################################################################################################
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_dim, 16)
        self.linear2 = nn.Linear(16, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, output_dim)


    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.linear4(x)


final_epsilon = 0.05
initial_epsilon = 1
epsilon_decay = 5000
global steps_done
steps_done = 0
times_count = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = final_epsilon + (initial_epsilon - final_epsilon) * math.exp(-1. * steps_done / epsilon_decay)
    if sample > eps_threshold:
        with torch.no_grad():
            state = torch.Tensor(state)
            steps_done += 1
            q_calc = model(state)
            node_activated = int(torch.argmax(q_calc))
            return node_activated
    else:
        node_activated = random.randint(0,1)
        steps_done += 1
        return node_activated


class ReplayMemory(object): # Stores [state, reward, action, next_state, done]

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [[],[],[],[],[]]
        self.memory2 =  [[],[],[],[],[]]

    def push(self, data):
        """Saves a transition."""
        for idx, point in enumerate(data):
            #print("Col {} appended {}".format(idx, point))
            self.memory[idx].append(point)
        self.memory2[0] = [data[0]]
        self.memory2[1] = [data[1]]
        self.memory2[2] = [data[2]]
        self.memory2[3] = [data[3]]
        self.memory2[4] = [data[4]]
        #print("self.memory2",data[1])


    def sample(self, batch_size):
        rows = random.sample(range(0, len(self.memory[0])), batch_size)
        experiences = [[],[],[],[],[]]
        for row in rows:
            for col in range(5):
                experiences[col].append(self.memory[col][row])
        return experiences

    def current_sample(self, batch_size):
        #print(self.memory2)
        return self.memory2

    def __len__(self):
        return len(self.memory[0])


input_dim, output_dim = 4, 2
model = DQN(input_dim, output_dim)
#target_net = DQN(input_dim, output_dim)
#target_net.load_state_dict(model.state_dict())
#target_net.eval()
tau = 0.0001#1000
discount = 0.99

learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

memory = ReplayMemory(65536)
BATCH_SIZE = 100


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return 0
    experiences = memory.sample(BATCH_SIZE)
    state_batch = torch.Tensor(experiences[0])
    action_batch = torch.LongTensor(experiences[1]).unsqueeze(1)
    reward_batch = torch.Tensor([experiences[2]])
    next_state_batch = torch.Tensor(experiences[3])
    done_batch = experiences[4]
    SA_values = model(state_batch).gather(1, action_batch)
    NSA_values = torch.zeros(BATCH_SIZE)

    for idx, next_state in enumerate(next_state_batch):
        if done_batch[idx] == True:
            NSA_values[idx] = -1
        else:
            with torch.no_grad():
                # .max in pytorch returns (values, idx), we only want vals
                NSA_values[idx] = model(next_state_batch[idx]).max(0)[0]

    label = (reward_batch + discount * NSA_values).t()
    loss = F.smooth_l1_loss(label,SA_values)
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    '''New Gradient'''
    with torch.no_grad():
        SA_values1 = model(state_batch).gather(1, action_batch)
    NSA_values1 = torch.zeros(BATCH_SIZE)
    for idx, next_state in enumerate(next_state_batch):
        if done_batch[idx] == True:
            NSA_values1[idx] = torch.tensor(-1.0, requires_grad=True)
            #next_state_q_vals1[idx] = model(next_state_batch[idx]).max(0)[0]
        else:
            # .max in pytorch returns (values, idx), we only want vals
            NSA_values1[idx] = model(next_state_batch[idx]).max(0)[0]
    #print("NSA_values1",NSA_values1)
    #print("SA_values1", SA_values1)
    label1 = (reward_batch + discount * NSA_values1).t()
    #print("New", better_pred1)
    loss1 = F.smooth_l1_loss(SA_values1,label1)
    optimizer.zero_grad()
    loss1.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss1

points = []
losspoints = []

#save_state = torch.load("models/DQN_target_11.pth")
#model.load_state_dict(save_state['state_dict'])
#optimizer.load_state_dict(save_state['optimizer'])



env = gym.make('CartPole-v0')

for i_episode in range(1200):
    observation = env.reset()
    episode_loss = 0
    '''Soft Update'''
    '''
    for target_param, local_param in zip(target_net.parameters(), model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
    '''
    ''' Hard Update'''
    '''
    if i_episode % tau == 0:
        target_net.load_state_dict(model.state_dict())
    '''


    for t in range(300):
        #env.render()
        state = observation
        action = select_action(observation)
        observation, reward, done, _ = env.step(action)
        times_count += 1
        if done:
            next_state = [0,0,0,0]
        else:
            next_state = observation
        memory.push([state, action, reward, next_state, done])
        episode_loss = episode_loss + float(optimize_model())
        if done:
            points.append((i_episode, t+1))
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            print("Avg Loss: ", episode_loss / (t+1))
            losspoints.append((i_episode, episode_loss / (t+1)))
            if (i_episode % 100 == 0):
                eps = final_epsilon + (initial_epsilon - final_epsilon) * \
                    math.exp(-1. * steps_done / epsilon_decay)
                print(eps)
            if ((i_episode+1) % 50001 == 0):
                save = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(save, "models/DQN_target_" + str(i_episode // 50000) + ".pth")
            break
env.close()




x = [coord[0]  for coord in points]
y = [coord[1] for coord in points]

x2 = [coord[0] for coord in losspoints]
y2 = [coord[1] for coord in losspoints]

#plt.plot(x, y)

with open('Data_fgdqn5.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Rewards", "Loss"])
    writer.writerows(zip(points, losspoints))

plt.plot(x2, y2)
print(x,y)
print("----------")
print(x2,y2)
plt.show()
plt.savefig('plot_fgdqn5.pdf')

