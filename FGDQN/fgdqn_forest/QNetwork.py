#######################################################################################################################
#                                                                                                                     #
#    Kishor Patil                                                                                                     #
#    @ Oct, 2020; Reinforcement Learning                                                                              #
#    Web page crawling; Whittle indices;                                                                              #
#    Deep learning Framework                                                                                          #
#    QNetwork.py                                                                                                      #
#######################################################################################################################


import torch
import torch.nn as nn
#import numpy as np
import torch.nn.functional as F

#device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
torch.autograd.set_detect_anomaly(True)


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
