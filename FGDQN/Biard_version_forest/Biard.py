#######################################################################################################################
#    Full Gradient DQN - Forest Management                                                                            #
#    @ Jan, 2021; Reinforcement Learning                                                                              #
#    Deep learning Framework                                                                                          #
#    Train the DQN Agent                                                                                              #
#######################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import torch
import csv
plt.ion()

import torch.nn as nn
#import numpy as np
import torch.nn.functional as F

#device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
torch.autograd.set_detect_anomaly(True)


from collections import namedtuple
import random


# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Buffer_size = int(1e5)  # Replay Buffer Size
Batch_size = 25  # minibatch size



#######################################################################################################################
#                                           Q-Network                                                                 #
#######################################################################################################################

class QNetwork(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, fc1_unit = 2000, fc2_unit = 8):
        """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()  ## calls __init__ method of nn.Module class
        #self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(2, fc1_unit)
        #self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.fc2 = nn.Linear(fc1_unit, 1)

    ######## Inputs: actions is a list of actions that can be chosen in this state. ######
    def forward(self, state_batch,action_batch):
        state_batch = state_batch.float()
        action_batch = action_batch.float()
        input_x = torch.stack((state_batch,action_batch),1).squeeze()
        #print("state_inuput",input_x)
        x_state = F.relu(self.fc1(input_x))
        #x_state = F.relu(self.fc2(x_state))
        actions = self.fc2(x_state)
        return actions

#######################################################################################################################
#                                       Replay Memory                                                                 #
#######################################################################################################################
random.seed(30)
Transition = namedtuple('Transition', ('state', 'action', 'reward','next_state'))


class ReplayMemory(object):

    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.memory2 = []
        self.position = 0
        self.is_full = False

    def Push_transition(self,arm_states, current_action, immediate_reward, next_state):
        ## Saves a transition into a Replay memory
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        #print(arm_states, current_action, immediate_reward, next_state)
        self.memory[self.position] = Transition(arm_states, current_action, immediate_reward, next_state)
        self.memory2 = [Transition(arm_states, current_action, immediate_reward, next_state)]
        #print(self.position)
        self.position = (self.position + 1) % self.capacity #Makes self.position zero once capacity is reached and starts replacing the oldest tuple

    def Sample_batch_train(self,batch_size):
        #select random batch to train the network
        #print(self.memory)
        return random.sample(self.memory,batch_size)
    def Return_Current_sample(self,batch_size):
        #print("memory2",self.memory2)
        return random.sample(self.memory2,1)

    def Sample_batch_FG(self,s1,a1):
        range_T = self.position if self.position < self.capacity else self.capacity
        print(range_T)
        output = []
        for i in range(range_T):
            j = 0
            if self.memory[i].state == s1 and self.memory[i].action==a1:
                output.append(self.memory[i])
                #output[j] = self.memory[i]
                j += 1
        return output

    def __len__(self):
        """" Return the current size of memory"""
        return len(self.memory)

    def print_memory(self):
        print(self.memory)

#######################################################################################################################
#                                       DQN Main                                                                      #
#######################################################################################################################
import math

import matplotlib
# from itertools import count
from collections import namedtuple
'''
## Import function approximator for Q from other file
from QNetwork import QNetwork
# ## Import Replay Memory
from ReplayMemory import ReplayMemory
'''
import torch.optim as optim
import torch.nn.functional as F

# import torchvision.transforms as T

#torch.autograd.set_detect_anomaly(True)
# Set up matplotlib

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

#plt.ion()

# if GPU is to be used
#device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

'''
Buffer_size = int(1e5)  # Replay Buffer Size
Batch_size = 2  # minibatch size
gamma = 0.99  # discount factor
lr = 0.01  # learning rate
'''

Eps_start = 0.7
Eps_end = 0.1
Eps_decay = 200
UPDATE_EVERY = 1
#learning_C = 0.000008
#TAU = 0.001
#Target_update = 100



# Intitialise the replay memory
memory = ReplayMemory(100000)


class DQN_agent:
    """Fixed -size buffe to store experience tuples."""
    def __init__(self, state_size=1, action_size=2, batch_size=3, exp_rate=0.3, seed=123):
        """ Initialize a ReplayBuffer object.
        ======
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.state_size = state_size
        self.batch_size = batch_size
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self.seed = np.random.seed(seed)
        self.exp_rate = exp_rate
        self.times = 0
        self.t_step = 0
        #self.Reward = np.zeros(self.state_size)
        self.state = torch.zeros(state_size,dtype=int)  # Record the state of each arm
        #self.losses = []
        self.gamma = 0.99 # Discount parameter
        self.p = 0.2
        # Initialises the Q- Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        #self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        #self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        #self.qnetwork_target.eval()


        # Define the Optimiser
        #self.optimizer = optim.SGD(self.qnetwork_local.parameters(), lr=0.001)
        #self.optimizer = optim.ASGD(self.qnetwork_local.parameters(), 0.01)
        self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=0.001)
        #self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.00001)

    def Utility(self,state,action):
        # Geometeric Utility
        # Action 0 - wait; 1 - Cut
        if action == 1:
            return state
        else:
            return torch.tensor([0])

    def ChooseAction(self, state):
        state = state.clone()
        #eps_threshold = Eps_start* (1 - 1/(self.times+1))
        eps_threshold =  Eps_end + (Eps_start - Eps_end) * math.exp(-1 * self.times / Eps_decay)
        self.times += 1
        # explore
        if np.random.uniform(0, 1) <= eps_threshold:
            action = torch.tensor([np.random.choice([0, 1])])
            #print("Random Choice", action)
        else:
            with torch.no_grad():
                Q0 = self.qnetwork_local(state, torch.tensor([0]))
                Q1 = self.qnetwork_local(state, torch.tensor([1]))
                #print("action to choose", Q0, Q1)
                action = torch.tensor([0.0]) if Q0 >= Q1 else torch.tensor([1.0])
        return action

    def ChooseAction_eval(self, state):
        state = state.clone()
        with torch.no_grad():
            Q0 = self.qnetwork_local(state, torch.tensor([0]))
            Q1 = self.qnetwork_local(state, torch.tensor([1]))
            #print("action to choose", Q0, Q1)
            action = torch.tensor([0.0]) if Q0 >= Q1 else torch.tensor([1.0])
        return action

    def Step(self, state, action):
        # extract current state, next state , action and reward
        state = state.float()
        current_action = action
        '''Current states'''
        current_state = state.clone()
        '''Evaluate immediate rewards and next states'''
        immediate_reward = self.Utility(current_state, current_action)
        # next_state = torch.zeros(1)
        if current_action == 1.0:
            next_state = torch.tensor([0])
        else:
            if np.random.uniform() < self.p:
                if current_state < (self.state_size - 1):
                    temp1 = int(10*current_state) + 1
                    print(temp1)
                    next_state = np.random.randint(temp1,size =1)/10
                    next_state = torch.tensor(next_state)
                #print("Next state increases", next_state)
            else:
                # print("nxt state",np.arange(state))
                next_state = current_state + 0.1 # torch.tensor([np.random.choice(np.arange(current_state))])

        '''Save experience in replay memory'''
        memory.Push_transition(current_state, current_action, immediate_reward, next_state)
        #memory.print_memory()
        loss = self.Optimise_model(current_state, current_action, immediate_reward, next_state)
        return loss, next_state, immediate_reward

    def Optimise_model(self,s1,a1,r1,n1):
        # Perform a single step of the optimisation
        if len(memory) < 400:
            #print('Transitions list is none')
            return 0
        # ----- To DO -------# Use experience replay to fetch samples with fixed (s1,a1)

        #Samples2 = memory.Sample_batch_FG(s1,a1)
        #Samples = random.sample(Samples2,1)
        Samples = [Transition(s1,a1,r1,n1)]
        #Samples1 = memory.Sample_batch_train(25)
        #print("Sample", len(Samples))
        # Transpose the batch. This converts batch-array of Transitions to Transition of batch-arrays.
        batch = self.Transition(*zip(*Samples))
        state_batch = torch.stack(batch.state,0)#
        action_batch = torch.stack(batch.action,0)#
        reward_batch = torch.stack(batch.reward,0)#
        next_state_batch = torch.stack(batch.next_state,0)#
        '''
        print("States batch input")
        print(state_batch)
        print("action batch input")
        print(action_batch)
        print("next state batch input")
        print(next_state_batch)
        '''
        #
        with torch.no_grad():
            len1 = action_batch.size()[0]
            action0 = torch.tensor([0]* len1).unsqueeze(1)
            action1 = torch.tensor([1]* len1).unsqueeze(1)
            NSA_values0 =  self.qnetwork_local(next_state_batch,action0)
            NSA_values1 = self.qnetwork_local(next_state_batch,action1)
            NSA_values = torch.max(NSA_values0,NSA_values1)
            #print(NSA_values.size())
            labels = reward_batch +  self.gamma * NSA_values
            #print(reward_batch.size(), NSA.size(),labels.size())
        SA_values = self.qnetwork_local(state_batch, action_batch)

        loss = F.mse_loss(labels, SA_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        First_Batch_Samples_Diff = (labels - SA_values).mean().detach()
        #print("First_Batch_Samples_Diff",First_Batch_Samples_Diff)

        '''New Gradient'''
        #Use Inbuilt optimiser
        #Generate new sample with s1 and a1

        if a1 == 1.0:
            n2 = torch.tensor([0])
        else:
            if np.random.uniform() < self.p:
                if s1 < (self.state_size - 1):
                    temp2 = int(10 * s1) +1
                    n2 = np.random.randint(temp2,size =1)/10
                    n2 = torch.tensor(n2)
                #print("Next state increases", next_state)
            else:
                # print("nxt state",np.arange(state))
                n2 = s1 + 0.1 # torch.tensor([np.random.choice(np.arange(current_state))])


        Samples1 = [Transition(s1,a1,r1,n2)]
        # print("Sample", len(Samples))
        # Transpose the batch. This converts batch-array of Transitions to Transition of batch-arrays.
        batch1 = self.Transition(*zip(*Samples1))
        state_batch1 = torch.stack(batch1.state, 0)  #
        action_batch1 = torch.stack(batch1.action, 0)  #
        reward_batch1 = torch.stack(batch1.reward, 0)  #
        next_state_batch1 = torch.stack(batch1.next_state, 0)  #

        len1 = action_batch1.size()[0]
        action0 = torch.tensor([0] * len1).unsqueeze(1)
        action1 = torch.tensor([1] * len1).unsqueeze(1)
        NSA_values0 = self.qnetwork_local(next_state_batch1, action0)
        NSA_values1 = self.qnetwork_local(next_state_batch1, action1)
        NSA_values = torch.max(NSA_values0, NSA_values1)
        # print(NSA_values.size())
        labels1 = reward_batch1 + self.gamma * NSA_values
        with torch.no_grad():
            SA_values1 = self.qnetwork_local(state_batch1, action_batch1)
        loss1 = F.mse_loss(labels1, SA_values1)
        new_diff = (labels1 - SA_values1).mean().detach()
        #new_diff1 = 1 if new_diff ==  0 else new_diff
        #print("New_Diff:{}, Old_Diff:{}".format(new_diff,First_Batch_Samples_Diff))

        loss1 = loss1 * First_Batch_Samples_Diff / new_diff
        print("loss1",loss1)

        self.optimizer.zero_grad()
        loss1.backward()
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


        '''
        NQ0 = self.qnetwork_local(n1, torch.tensor([0]))
        NQ1 = self.qnetwork_local(n1, torch.tensor([1]))
        NBest =  r1 + self.gamma * torch.max(NQ0, NQ1)
        with torch.no_grad():
            QS_Current = self.qnetwork_local(s1, a1)
        # action_best = torch.tensor([0.0]) if NQ0>NQ1 else torch.tensor([1.0])
        loss1 = F.mse_loss(NBest, QS_Current)
        self.optimizer.zero_grad()
        #print("Loss1",loss1)
        loss1.backward()
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        '''

        # Bellman loss for current iteration
        with torch.no_grad():
            NQ0 = self.qnetwork_local(n1, torch.tensor([0]))
            NQ1 = self.qnetwork_local(n1, torch.tensor([1]))
            QS_Current = self.qnetwork_local(s1,a1)
            #action_best = torch.tensor([0.0]) if NQ0>NQ1 else torch.tensor([1.0])
            loss_itr = r1 + self.gamma *  torch.max(NQ0,NQ1) - QS_Current

        loss_itr = loss_itr.item()
        print("loss_itr", loss_itr * loss_itr)


        return loss_itr*loss_itr #* self.batch_size


#######################################################################################################################
#                                       Run DQN Agent Main loop                                                       #
#######################################################################################################################

''' 
Initialise the problem   
'''
state_size = 10
action_size = 2
seed = 12498

epsilon1 = 0

#PATH = "M_parameters.pt"

Agent = DQN_agent(state_size,action_size,Batch_size,epsilon1,seed)
optimal_policy = torch.tensor([0,0,0,1,1,1,1,1,1,1])  # Calculate it offline using Policy iteration
out_actions = torch.zeros(state_size)
Reward_points = []
loss_points = []
losses = []
Hamming_distance = []
Policy_at_each_itr = []
state_record = []
def dqn_train(Episodes = 1,iterations = 30000):
    Reward_Scores = []
    #eps = Eps_start
    #lambdaW = torch.zeros(Episodes,iterations,Arms,state_size)

    for i in range(Episodes):
        episode_loss = 0
        #score = 0
        #scores = []
        for t in range(iterations):
            state = torch.zeros(1, dtype=int)
            for s in range(state_size):
                for a in range(action_size):
                    current_action = torch.tensor([a])
                    '''
                    Perform the current action for one step
                    Optimise_model() is called inside the Agent.Step()
                    Thus automatically performing Qnetwork and everything else            
                    '''
                    loss, next_state,reward = Agent.Step(state,current_action)
                    losses.append(loss)
                    Policy_actions = torch.zeros(state_size)
                    for j in range(10):
                        Policy_actions[j] = Agent.ChooseAction_eval(torch.tensor([j / 10]))
                    state_record.append(state.squeeze().numpy())
                    Policy_at_each_itr.append(Policy_actions.numpy())
                    Hamming_distance.append(sum(a != b for a, b in zip(Policy_actions, optimal_policy)).numpy())

                #print("state {}, action{}, next_state{}, Reward{}".format(state,current_action,next_state,reward))
                    #print("loss",loss)
                    #print("-----------------------")
                state = state+0.1
    ## evaluate

    #state = torch.zeros(1, dtype=int)
    for j in range(10):
        out_actions[j] = Agent.ChooseAction_eval(torch.tensor([j/10]))
    print(out_actions)

dqn_train()



with open('Data9.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Loss", "states","Policy"])
    writer.writerows(zip(losses, state_record,Policy_at_each_itr))

avg_loss = []
for i in range(len(losses) - 700):
    avg_loss.append(np.mean(losses[500+i:700+i]))

# plot the scores
fig1 = plt.figure()
#plt.plot(x, y)
plt.plot(avg_loss)
plt.show()
plt.xlabel('Itr #')
plt.ylabel('loss ')
#plt.ylim(0,20)
plt.savefig('error_fgdqn9.pdf')

fig2 = plt.figure()
#plt.plot(x, y)
plt.plot(out_actions)
plt.show()
plt.xlabel('Optimalaction #')
plt.ylabel('states ')
#plt.ylim(0,20)
plt.savefig('states_actions9.pdf')


fig2 = plt.figure()
#plt.plot(x, y)
plt.plot(Hamming_distance[600:])
plt.show()
plt.xlabel('Itr #')
plt.ylabel('Hammin distance ')
#plt.ylim(0,20)
plt.savefig('hamming9.pdf')

