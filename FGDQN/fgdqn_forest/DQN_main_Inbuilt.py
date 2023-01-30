#######################################################################################################################
#                                                                                                                     #
#    Kishor Patil                                                                                                     #
#    @ Oct, 2020; Reinforcement Learning                                                                              #
#    Web page crawling; Whittle indices;                                                                              #
#    Deep learning Framework                                                                                          #
#    DQN Main python File                                                                                             #
#######################################################################################################################

import math
# import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from itertools import count
from collections import namedtuple

## Import function approximator for Q from other file
from QNetwork import QNetwork

# ## Import Replay Memory
from ReplayMemory import ReplayMemory

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# import torchvision.transforms as T

torch.autograd.set_detect_anomaly(True)
# Set up matplotlib

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
'''
Buffer_size = int(1e5)  # Replay Buffer Size
Batch_size = 2  # minibatch size
gamma = 0.99  # discount factor
lr = 0.01  # learning rate
'''
Eps_start = 0
Eps_end = 0
Eps_decay = 200
UPDATE_EVERY = 1
#learning_C = 0.000008
TAU = 0.001
Target_update = 100
# psi_1_1s = []
# psi_2_1s = []


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
        self.gamma = 0.98 # Discount parameter
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
        Samples = memory.Sample_batch_FG(s1,a1)
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

        ## Next states from target network
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
        print("Length",len1)
        loss = F.mse_loss(labels, SA_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.qnetwork_local.parameters():
             param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        '''New Gradient'''
        '''
        len1 = action_batch.size()[0]
        action0 = torch.tensor([0]* len1).unsqueeze(1)
        action1 = torch.tensor([1]* len1).unsqueeze(1)
        NSA_values0 =  self.qnetwork_local(next_state_batch,action0)
        NSA_values1 = self.qnetwork_local(next_state_batch,action1)
        NSA_values = torch.max(NSA_values0,NSA_values1)
        #print(NSA_values.size())
        labels1 = reward_batch +  self.gamma * NSA_values
        with torch.no_grad():
            SA_values1 = self.qnetwork_local(state_batch, action_batch)
        loss1 = F.mse_loss(labels1, SA_values1)
        self.optimizer.zero_grad()
        loss1.backward()
        for param in self.qnetwork_local.parameters():
             param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        '''
        with torch.no_grad():
            len1 = action_batch.size()[0]
            action0 = torch.tensor([0] * len1).unsqueeze(1)
            action1 = torch.tensor([1] * len1).unsqueeze(1)
            NSA_values0 = self.qnetwork_local(next_state_batch, action0)
            NSA_values1 = self.qnetwork_local(next_state_batch, action1)
            NSA_values = torch.max(NSA_values0, NSA_values1)
            # print(NSA_values.size())
            labels1 = reward_batch + self.gamma * NSA_values
            SA_values1 = self.qnetwork_local(state_batch, action_batch)
            FIRST_DIFF = (labels1 - SA_values1).mean().detach()

        '''New Gradient'''
        # use Inbuilt Optimiser
        NQ0 = self.qnetwork_local(n1, torch.tensor([0]))
        NQ1 = self.qnetwork_local(n1, torch.tensor([1]))
        NBest =  r1 + self.gamma * torch.max(NQ0, NQ1)
        with torch.no_grad():
            QS_Current = self.qnetwork_local(s1, a1)
        # action_best = torch.tensor([0.0]) if NQ0>NQ1 else torch.tensor([1.0])
        New_Diff = (NBest - QS_Current).mean().detach()
        loss1 = F.mse_loss(NBest, QS_Current)
        new_diff1 = 1 if New_Diff == 0 else New_Diff

        loss1 = loss1 * FIRST_DIFF / new_diff1

        self.optimizer.zero_grad()
        #print("Loss1",loss1)
        loss1.backward()
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


        #self.losses.append(loss)
        # Bellman loss
        with torch.no_grad():
            NQ0 = self.qnetwork_local(n1, torch.tensor([0]))
            NQ1 = self.qnetwork_local(n1, torch.tensor([1]))
            QS_Current = self.qnetwork_local(s1,a1)
            #action_best = torch.tensor([0.0]) if NQ0>NQ1 else torch.tensor([1.0])
            loss_itr = r1 + self.gamma *  torch.max(NQ0,NQ1) - QS_Current

        loss_itr = loss_itr.item()
        print("loss_itr", loss_itr * loss_itr)

        #self.qnetwork_local.zero_grad()
        #Grad_Q.backward()




        # ------------------- update target network ------------------- #
        #if self.times % Target_update == 0:
        #    self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        #self.soft_update(TAU)

        return loss_itr*loss_itr #* self.batch_size

        #self.optimizer.step()



    def update_param(self,param, grad, learning_rate):
        return param + learning_rate * grad

    def soft_update(self, tau):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

