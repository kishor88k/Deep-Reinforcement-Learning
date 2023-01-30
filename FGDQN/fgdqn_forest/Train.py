#######################################################################################################################
#                                                                                                                     #
#    Kishor Patil                                                                                                     #
#    @ Oct, 2020; Reinforcement Learning                                                                              #
#    Web page crawling; Whittle indices;                                                                              #
#    Deep learning Framework                                                                                          #
#    Train the DQN Agent                                                                                              #
#######################################################################################################################


import matplotlib.pyplot as plt
import numpy as np
from DQN_main_Inbuilt import DQN_agent
import torch
import csv

plt.ion()

# if GPU is to be used

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Buffer_size = int(1e5)  # Replay Buffer Size
Batch_size = 25  # minibatch size


''' 
Initialise the problem   
'''
state_size = 10
action_size = 2
seed = 12498

epsilon1 = 0
TAU = 0.001

#PATH = "M_parameters.pt"


Agent = DQN_agent(state_size,action_size,Batch_size,epsilon1,seed)

optimal_policy = torch.tensor([0,0,0,1,1,1,1,1,1,1])
out_actions = torch.zeros(state_size)
Reward_points = []
loss_points = []
losses = []
Hamming_distance = []
Policy_at_each_itr = []
state_record = []
def dqn_train(Episodes = 1,iterations = 2000):
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



with open('Data19.csv', "w") as f:
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
plt.savefig('error_fgdqn19.pdf')

fig2 = plt.figure()
#plt.plot(x, y)
plt.plot(out_actions)
plt.show()
plt.xlabel('Optimalaction #')
plt.ylabel('states ')
#plt.ylim(0,20)
plt.savefig('states_actions19.pdf')

fig2 = plt.figure()
#plt.plot(x, y)
plt.plot(Hamming_distance[600:])
plt.show()
plt.xlabel('Itr #')
plt.ylabel('Hamming_distance ')
#plt.ylim(0,20)
plt.savefig('hamming19.pdf')


