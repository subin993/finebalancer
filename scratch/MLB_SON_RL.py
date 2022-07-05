import random
import gym
import numpy as np
import math 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque 
from ns3gym import ns3env
import matplotlib.pyplot as plt

EPISODES = 200
port=1122 # Should be consistent with NS-3 simulation port
stepTime=0.2
startSim=0
Usersnum=41
seed=3
batch_size = 32
simArgs = {}
debug=True
step_CIO=3 # CIO value step in the discrete set {-6, -3, 0, 3, 6}
Result_row=[]
Rew_ActIndx=[]
cio_action = [0,0,0,0,0]

max_env_steps = 50  #Maximum number of steps in every episode
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1
        self.epsilon_end = 0.01
        self.epsilon_decay = 200
        self.steps_done = 0
        self.learning_rate = 0.001
        # self.Prev_Mean=0 #initial mean of the targets (for target normalization)
        # self.Prev_std=1 #initial std of the targets (for target normalization)
        self.model = self._build_model()
        # self.target_model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        # self.update_target_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24,24),
            nn.ReLU(),
            nn.Linear(24,self.action_size)
        )
        return model
    
    # def update_target_model(self):
    #     # copy weights from the CIO selection network to target network
    #     self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, torch.FloatTensor([reward]), torch.FloatTensor([next_state])))
    
    def act(self, state):
        eps_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        act_values = self.model(state)
        # print(type(act_values))
        # print("Predicted action for this state is: {}".format(np.argmax(act_values[0])))
        if np.random.rand() <= eps_threshold:
            return torch.LongTensor([random.randrange(self.action_size)]) # [[0]] or [[1]] or [[2]]
        # this part is not perfect
        return act_values.data.max(1)[1].view(1,-1)  # returns action index (0 : CIO -3 or 1 : CIO 0 or 2 : CIO +3)
    
    def learn(self):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size) # [(state,action,reward,next_state,done), (~), ..,  ] : tuple in list
        states, actions, rewards, next_states = zip(*batch) # states = (state 1, state 2,..,state batch_size) : tuple

        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)

        current_q = self.model(states).gather(1,actions)
        max_next_q = self.model(next_states).detach().max(1)[0]
        expected_q = rewards + (self.gamma * max_next_q)

        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
################################################################


if __name__ == "__main__" :
    env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
    env._max_epsiode_steps = max_env_steps
    ac_space = env.action_space # Getting the action space
    state_size = 5 # Num of observation
    action_size = 3

    agent1 = DQNAgent(state_size, action_size)
    agent2 = DQNAgent(state_size, action_size)
    agent3 = DQNAgent(state_size, action_size)
    agent4 = DQNAgent(state_size, action_size)
    agent5 = DQNAgent(state_size, action_size)

    done = False
    batch_size = 32
    reward_sum = []

    for e in range(EPISODES):
        state = env.reset()
        state1 = np.reshape(state['rbUtil'], [5,1])
        R_rewards = np.reshape(state['rewards'], [5,1])
        R_rewards = [j for sub in R_rewards for j in sub]

        state = state1
        state = np.reshape(state, [1,state_size])
   
        for time in range(max_env_steps):
            print("*******************************")

            print("episode: {}/{}, step: {}".format(e+1, EPISODES, time))
            state = torch.FloatTensor([state])

            action1 = agent1.act(state)
            action2 = agent2.act(state)
            action3 = agent3.act(state)
            action4 = agent4.act(state)
            action5 = agent5.act(state)

            # For many action error
            # action1 = torch.tensor([[action1[0][0]]])
            # action2 = torch.tensor([[action2[0][0]]])
            # action3 = torch.tensor([[action3[0][0]]])
            # action4 = torch.tensor([[action4[0][0]]])
            # action5 = torch.tensor([[action5[0][0]]])

            print('action1 : ',action1)
            print('action2 : ',action2)
            print('action3 : ',action3)
            print('action4 : ',action4)
            print('action5 : ',action5)

            action = []
            action.append(action1.item())
            action.append(action2.item())
            action.append(action3.item())
            action.append(action4.item())
            action.append(action5.item())

            idx = 0
            for i in action :
                if(i == 0) : cio_action[i] -= 3
                elif(i == 2) : cio_action[i] += 3
                idx += 1

            next_state, reward, done, _ = env.step(cio_action)
            
            if next_state is None:
                EPISODES = EPISODES+1
                break
            
            state1 = np.reshape(next_state['rbUtil'], [1,5])
            R_rewards = np.reshape(next_state['rewards'], [5,1])
            R_rewards = [j for sub in R_rewards for j in sub]
            
            sum = 0
            for i in R_rewards :
                sum = sum + i
            # reward_sum.append(sum/5)
            print("Average Throughput : ", sum/5)

            next_state = state1

            agent1.remember(state, action1, R_rewards[0], next_state)
            agent2.remember(state, action2, R_rewards[1], next_state)
            agent3.remember(state, action3, R_rewards[2], next_state)
            agent4.remember(state, action4, R_rewards[3], next_state)
            agent5.remember(state, action5, R_rewards[4], next_state)

            state = next_state

            agent1.learn()
            agent2.learn()
            agent3.learn()
            agent4.learn()
            agent5.learn()



                