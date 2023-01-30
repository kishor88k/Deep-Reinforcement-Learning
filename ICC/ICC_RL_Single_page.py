#######################################################################################################################
#    Full Gradient DQN - Forest Management                                                                            #
#    @ May, 2021; Reinforcement Learning : Web Crawling                                                                             #
#    Deep learning Framework                                                                                          #
#    Train the DQN Agent                                                                                              #
#######################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import random
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
    def __init__(self, state_size, action_size, seed, fc1_unit = 4000, fc2_unit = 8, fc3_unit = 16):
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
        self.fc1 = nn.Linear(state_size, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        #self.fc3 = nn.Linear(fc2_unit, fc3_unit)
        self.fc4 = nn.Linear(fc2_unit, 4)

    ######## Inputs: actions is a list of actions that can be chosen in this state. ######
    def forward(self, state_batch):
        state_batch = state_batch.float()
        #action_batch = action_batch.float()
        #input_x = torch.stack((state_batch,action_batch),1).squeeze()
        #print("state_inuput",input_x)
        x_state = F.relu(self.fc1(state_batch))
        x_state = F.relu(self.fc2(x_state))
        #x_state = F.relu(self.fc3(x_state))
        actions = self.fc4(x_state)
        return actions



#######################################################################################################################
#                                       Replay Memory                                                                 #
#######################################################################################################################
random.seed(30)
Transition = namedtuple('Transition', ('state', 'action', 'reward','next_state'))


class ReplayMemory(object): # Stores [state, action, reward, next_state]

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [[],[],[],[]]
        self.memory2 =  [[],[],[],[]]

    def push(self, data):
        """Saves a transition."""
        for idx, point in enumerate(data):
            #print("Col {} appended {}".format(idx, point))
            self.memory[idx].append(point)
        self.memory2[0] = [data[0]]
        self.memory2[1] = [data[1]]
        self.memory2[2] = [data[2]]
        self.memory2[3] = [data[3]]
        #print("self.memory2",data[1])

    def sample(self, batch_size):
        rows = random.sample(range(0, len(self.memory[0])), batch_size)
        experiences = [[],[],[],[]]
        for row in rows:
            for col in range(4):
                experiences[col].append(self.memory[col][row])
        return experiences

    def current_sample(self, batch_size):
        #print(self.memory2)
        return self.memory2

    def __len__(self):
        return len(self.memory[0])


import math

import matplotlib

import torch.optim as optim

# import torchvision.transforms as T

# torch.autograd.set_detect_anomaly(True)
# Set up matplotlib

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

Eps_start = 0.7
Eps_end = 0.1#0.1
Eps_decay = 200
UPDATE_EVERY = 1

# learning_C = 0.000008
# TAU = 0.001
# Target_update = 100

# Intitialise the replay memory
memory = ReplayMemory(100000)


class DQN_agent:
    """Fixed -size buffe to store experience tuples."""

    def __init__(self, state_size=1, action_size=2, batch_size=3, exp_rate=0.3, mu=[5, 0.2, 1, 0.5], seed=123):
        """ Initialize a ReplayBuffer object.
        ======
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.Num_of_arms = 4
        self.Max_age = 20
        self.mu = mu
        self.D = [0, 0, 0, 0]
        self.action_size = action_size
        self.state_size = state_size
        self.batch_size = batch_size
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self.seed = np.random.seed(seed)
        self.exp_rate = exp_rate
        self.times = 0
        self.t_step = 0
        # self.Reward = np.zeros(self.state_size)
        self.state = torch.zeros(state_size, dtype=int)  # Record the state of each arm
        # self.losses = []
        self.gamma = 0.95  # Discount parameter
        self.p = 0.05
        # Initialises the Q- Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)

        # Define the Optimiser
        # self.optimizer = optim.SGD(self.qnetwork_local.parameters(), lr=0.001)
        # self.optimizer = optim.ASGD(self.qnetwork_local.parameters(), 0.01)
        self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=0.001)
        # self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.00001)

    def Utility(self, state, Y,action):
        # extract age and page change indicators from states
        Y = torch.tensor(Y)
        # find the arm being selected
        arm_index = action
        # (action==1).nonzero().item()

        # check the corresponding page has been changed or not
        Ind_Page_Change = Y[arm_index]
        # Return the

        return self.mu[arm_index] * Ind_Page_Change

    def ChooseAction(self, state, Y):
        # state = state.clone()
        # eps_threshold = Eps_start* (1 - 1/(self.times+1))
        eps_threshold = Eps_end + (Eps_start - Eps_end) * math.exp(-1 * self.times / Eps_decay)
        self.times += 1
        if self.times > 20000:
            eps_threshold = 0.001
        # explore
        if np.random.uniform(0, 1) <= eps_threshold:
            action = torch.tensor([np.random.choice([0, 1, 2, 3])])
            # print("Random Choice", action)
        else:
            # Make the state input to neural network
            state[4:] = torch.tensor(Y)
            with torch.no_grad():
                Q_calc = self.qnetwork_local(state)
                # print("Kishor",Q_calc)
                pred_probab = nn.Softmax(dim=-1)(Q_calc)
                # print("action to choose", Q0, Q1)
                action = pred_probab.argmax().unsqueeze(0)
                # action = torch.argmax(Q_calc)
        #Save model
        if self.times == 14500:
            PATH = "DRL.pth"
            save = {'state_dict': Agent.qnetwork_local.state_dict(), 'optimizer': Agent.optimizer.state_dict()}
            torch.save(save, PATH)
        return action

    '''
    def ChooseAction_eval(self, state):
        state = state.clone()
        with torch.no_grad():
            Q0 = self.qnetwork_local(state, torch.tensor([0]))
            Q1 = self.qnetwork_local(state, torch.tensor([1]))
            #print("action to choose", Q0, Q1)
            action = torch.tensor([0.0]) if Q0 >= Q1 else torch.tensor([1.0])
        return action
    '''

    def Step(self, state, Y, action):
        # extract current state, next state , action and reward
        # state = state.float()
        current_action = action
        '''Current states'''
        current_state = state
        #current_state[self.Num_of_arms:] = torch.tensor(Y)
        '''Evaluate immediate rewards and next states'''
        # immediate_reward = np.zeros(self.Num_of_arms)
        immediate_reward = self.Utility(current_state, Y, current_action)

        # Next State

        next_state = torch.zeros(self.Num_of_arms * 2, dtype=int)

        ###Age Update
        for i in range(self.Num_of_arms):
            if i == action:
                next_state[i] = 0
            else:
                next_state[i] = state[i] + 1

            '''
            if state[i] < self.Max_age -1:
                next_state[i] = (1 - current_action[i]) * (state[i] + 1)
            else:
                next_state[i] = (1 - current_action[i]) * state[i]
            '''
            ### Page change indicators for next state
            # Call the neural network and apply current action


        # for i in range(self.Num_of_arms: self.Num_of_arms*2):
        next_state[self.Num_of_arms:] = torch.tensor(Y)

        '''Save experience in replay memory'''
        memory.push([current_state, current_action, immediate_reward, next_state])
        # memory.print_memory()
        loss = self.Optimise_model(current_state, current_action, immediate_reward, next_state)
        return loss, next_state, immediate_reward

    def Optimise_model(self, s1, a1, r1, n1):
        # Perform a single step of the optimisation
        if len(memory) < 400:
            # print('Transitions list is none')
            return 0
        # ----- To DO -------# Use experience replay to fetch samples with fixed (s1,a1)
        experiences = memory.sample(self.batch_size)
        # print("Sample", len(Samples))
        # Transpose the batch. This converts batch-array of Transitions to Transition of batch-arrays.
        # print("Experiences", experiences)
        state_batch = torch.stack(experiences[0], 0)  # torch.Tensor(experiences[0])
        action_batch = torch.stack(experiences[1], 0)  # torch.LongTensor(experiences[1]).unsqueeze(1)
        reward_batch = torch.stack(experiences[2], 0)  # torch.Tensor([experiences[2]])
        next_state_batch = torch.stack(experiences[3], 0)  # torch.Tensor(experiences[3])

        '''
        print("States batch input")
        print(state_batch)
        print("action batch input")
        print(action_batch)
        print("next state batch input")
        print(next_state_batch)
        '''

        SA_values = self.qnetwork_local(state_batch).gather(1, action_batch).squeeze()

        # print("SA_Values",SA_values)

        NSA_values = torch.zeros(self.batch_size)

        for idx, next_state in enumerate(next_state_batch):
            with torch.no_grad():
                # .max in pytorch returns (values, idx), we only want vals
                NSA_values[idx] = self.qnetwork_local(next_state_batch[idx]).max(0)[0]

        label = (reward_batch.squeeze() + self.gamma * NSA_values).t()
        loss = F.mse_loss(label, SA_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        '''New Gradient'''
        # use Inbuilt Optimiser
        with torch.no_grad():
            SA_values1 = self.qnetwork_local(state_batch).gather(1, action_batch).squeeze()
        NSA_values1 = torch.zeros(self.batch_size)
        for idx, next_state in enumerate(next_state_batch):
            # .max in pytorch returns (values, idx), we only want vals
            NSA_values1[idx] = self.qnetwork_local(next_state_batch[idx]).max(0)[0]
        # print("NSA_values1",NSA_values1)
        # print("SA_values1", SA_values1)
        label1 = (reward_batch.squeeze() + self.gamma * NSA_values1).t()
        # print("New", better_pred1)
        loss1 = F.smooth_l1_loss(SA_values1, label1)
        self.optimizer.zero_grad()
        loss1.backward()
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss1.data.numpy()


#######################################################################################################################
#                                       Run DQN Agent Main loop                                                       #
#######################################################################################################################

''' 
Initialise the problem   
'''

####
state_size = 8
action_size = 2
seed = 12498

epsilon1 = 0

# PATH = "M_parameters.pt"

mu = [5, 0.2, 1, 0.5]

Agent = DQN_agent(state_size, action_size, Batch_size, epsilon1, mu, seed)

out_actions = torch.zeros(state_size)
Rewards = []
loss_points = []
losses = []
Avg_Rewards = []

Hamming_distance = []
Policy_at_each_itr = []
state_record = []


def dqn_train(Episodes=1, iterations=10000):
    Reward_Scores = []
    '''Define the Bandit Framework'''
    # Number of pages in each group
    Num_pages = [1,1,1,1]#[5, 80, 10, 5]
    Total_reward = 0
    # Importance of each group

    mu = [5, 0.2, 1, 0.5] # [5, 0.2, 1, 0.5 ]
    #
    ''' Intialise the pointers for each group'''
    G_pointer = np.zeros(4,dtype=int)

    '''Indicator if the next to be selectd page from the group has been changed or not'''
    Y = [0, 0, 0, 0]
    ## The change rates

    lambda1 = 0.6#0.1#0.9162#
    lambda3 = 0.08#0.002#0.08#
    lambda2 = 0.01#0.005# 0.01#
    lambda4 = 0.3#0.05 #0.35# #0.99

    ### Number of time slot since a particular page has been visited

    TS1 = np.ones(Num_pages[0])
    TS2 = np.ones(Num_pages[1])
    TS3 = np.ones(Num_pages[2])
    TS4 = np.ones(Num_pages[3])

    ##### Group page pointer

    Group_page_id_TS = [0, 0, 0, 0]
    for i in range(Episodes):
        episode_loss = 0
        # score = 0
        # scores = []
        state = torch.zeros(8, dtype=int)
        for t in range(iterations):
            # check if the corresponding page has been updated in the given time slot?
            current_action = Agent.ChooseAction(state, Y) #, Y = [0,0,1,0], # 1

            # Update Y_k

            # Take the pages from each group next to be selected and check if it has been changed since the last time it was visited
            Group_page_id_TS[0] = TS1[G_pointer[0]]
            Group_page_id_TS[1] = TS2[G_pointer[1]]
            Group_page_id_TS[2] = TS3[G_pointer[2]]
            Group_page_id_TS[3] = TS4[G_pointer[3]]

            # change the corresponding Y only : case:{}
            # Also Make the the corresponding page fresh i.e., set time slot to zero
            # Increase the pointer to indicate the next page to be selected in round-robin fashion
            # increase the pointer and change Y only when group is chosen
            if current_action == 0:
                Y[0] = 1 if random.random() < 1 - np.exp(-lambda1 * Group_page_id_TS[0]) else 0
                TS1[G_pointer[0]] = 0
                if G_pointer[0] < Num_pages[0] - 1:
                    G_pointer[0] = G_pointer[0] + 1
                else:
                    G_pointer[0] = 0
            elif current_action == 1:
                Y[1] = 1 if random.random() < 1 - np.exp(-lambda2 * Group_page_id_TS[1]) else 0
                TS2[G_pointer[1]] = 0
                if G_pointer[1] < Num_pages[1] - 1:
                    G_pointer[1] = G_pointer[1] + 1
                else:
                    G_pointer[1] = 0
            elif current_action == 2:
                Y[2] = 1 if random.random() < 1 - np.exp(-lambda3 * Group_page_id_TS[2]) else 0
                TS3[G_pointer[2]] = 0
                if G_pointer[2] < Num_pages[2] - 1:
                    G_pointer[2] = G_pointer[2] + 1
                else:
                    G_pointer[2] = 0
            else:
                Y[3] = 1 if random.random() < 1 - np.exp(-lambda4 * Group_page_id_TS[3]) else 0
                TS4[G_pointer[3]] = 0
                if G_pointer[3] < Num_pages[3] - 1:
                    G_pointer[3] = G_pointer[3] + 1
                else:
                    G_pointer[3] = 0

            '''
                        Perform the current action for one step
                        Optimise_model() is called inside the Agent.Step()
                        Thus automatically performing Qnetwork and everything else            
             '''
            loss, next_state, reward = Agent.Step(state, Y, current_action)
            losses.append(loss)
            Rewards.append(reward.squeeze().numpy())
            Total_reward += reward.squeeze().numpy()
            Avg_Rewards.append(Total_reward / t)
            print("loss = {}, Avg_Reward = {} Reward = {}".format(loss, Avg_Rewards[t],reward.squeeze().numpy()))

            #Y2 = [0,1,1,0]
            #state = [X,Y], X = [1,1,0,1], X1 = [2,0,1,2]

            # Increase the time slot for each page since the last visit
            TS1 = TS1 + 1
            TS2 = TS2 + 1
            TS3 = TS3 + 1
            TS4 = TS4 + 1

            print("Iteration {}, state {}, action{},next_state{}, Reward{}".format(t,state,current_action,next_state,reward))
            print("TS1{},TS2{},TS3{},TS4{}".format(TS1,TS2,TS3,TS4))
            # print("loss",loss)
            # print("-----------------------")
            state = next_state
dqn_train()

############# Save the Result##########
with open('Data8_DRL.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Loss", "Rewards", "Avg_reward"])
    writer.writerows(zip(losses, Rewards, Avg_Rewards))

'''
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
plt.savefig('error_fgdqn1.pdf')
'''

