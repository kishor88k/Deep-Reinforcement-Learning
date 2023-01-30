#######################################################################################################################
#                                                                                                                     #
#    Kishor Patil                                                                                                     #
#    @ Oct, 2020; Reinforcement Learning                                                                              #
#    Web page crawling; Whittle indices;                                                                              #
#    Deep learning Framework                                                                                          #
#    Replay Memory #only 1 sample at the moment from replay memory                                                    #
#######################################################################################################################

from collections import namedtuple
import random
import torch
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

