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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPISODES = 10000 
max_env_steps = 120  # Maximum number of steps in every episode
port=1150
stepTime=0.5
startSim=0
seed=3
simArgs = {}
debug=True
cio_action = [0,0,0,0,0] # Initial CIO action
action_prev = [] 

# DDQN Agent Class
class DDQNAgent:
    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=20000)

        self.gamma = 0.95  
        self.epsilon = 1
        self.epsilon_end = 0.000000000000001
        self.epsilon_decay = 0.9999
        self.steps_done = 0
        self.learning_rate = 0.001

        self.model = self._build_model()
        self.target_model = self._build_model()

        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)

        self.update_target_model()

        self.loss = 0

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24,24),
            nn.ReLU(),
            nn.Linear(24,self.action_size)
        )
        return model.to(device)
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, torch.FloatTensor([reward]).to(device), torch.FloatTensor([next_state]).to(device)))
        if (self.epsilon > self.epsilon_end) :
            self.epsilon *= self.epsilon_decay
    
    def act(self, state):
        # Epsilon-greedy
        eps_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        
        act_values = self.model(state)

        if np.random.rand() <= eps_threshold:
            return torch.LongTensor([random.randrange(self.action_size)]).to(device) 

        return torch.argmax(act_values[0]).unsqueeze(0).to(device)  
    
    def learn(self):
        
        if len(self.memory) < 2 * batch_size:
            return

        batch = random.sample(self.memory, batch_size) 
        states, actions, rewards, next_states = zip(*batch) 

        states = torch.cat(states).squeeze()

        actions = torch.cat(actions).unsqueeze(1)

        rewards = torch.cat(rewards).unsqueeze(1)
        
        next_states = torch.cat(next_states).squeeze()

        current_q = self.model(states).gather(1,actions) 
        max_action = torch.argmax(self.model(next_states),dim=1).unsqueeze(0)
        max_actions = max_action.transpose(0,1)
        max_next_q = self.target_model(next_states).gather(1,max_actions)
        expected_q = rewards + (self.gamma * max_next_q)

        loss = F.mse_loss(current_q, expected_q)

        self.loss = loss
        print("Agent Loss: ",loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Main Code
if __name__ == "__main__" :
    env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
    env._max_epsiode_steps = max_env_steps
    ac_space = env.action_space
    
    # Number of states, actions
    state_size = 12 
    action_size = 5

    agent3 = DDQNAgent(state_size, action_size)
    agent4 = DDQNAgent(state_size, action_size)
    agent5 = DDQNAgent(state_size, action_size)

    done = False

    batch_size = 32

    break_ep = 0

    FullStep_Ep = []

    for e in range(EPISODES):
        
        state = env.reset()
        state1 = np.reshape(state['rbUtil'], [5,1])
        state1 = state1[2:5,0:1]

        MCSVec = np.array(state['MCSPen'])
        state2 = np.sum(MCSVec[:,:10], axis=1)
        state2 = np.reshape(state2, [5,1])
        state2 = state2[2:5,0:1]

        state3 = np.reshape(state['ServedUes'], [5,1])
        state3 = state3[2:5,0:1]

        state4 = np.reshape(state['Throughput'], [5,1])
        state4 = state4[2:5,0:1]

        R_rewards = np.reshape(state['Throughput'], [5,1])
        R_rewards = R_rewards[2:5,0:1]
        R_rewards = [j for sub in R_rewards for j in sub]

        state = np.concatenate( (state1, state2, state3, state4), axis=None )
        state = np.reshape(state, [1,state_size])

        for time in range(max_env_steps): 
            print("*******************************")
            print("episode: {}/{}, step: {}".format(e+1, EPISODES, time))

            state = torch.FloatTensor([state]).to(device)
            
            action1 = 6
            action2 = 6
            action3 = agent3.act(state)
            action4 = agent4.act(state)
            action5 = agent5.act(state)

            action = []
            action.append(6)
            action.append(6)
            action.append(action3.item())
            action.append(action4.item())
            action.append(action5.item())

            cio_action = [] 
            
            for i in action :
                if(i == 0) : 
                    cio_action.append(-6)
                elif(i == 1) : 
                    cio_action.append(-5)
                elif(i == 2) : 
                    cio_action.append(-4)
                elif(i == 3) : 
                    cio_action.append(-3)
                elif(i == 4) : 
                    cio_action.append(-2)
                elif(i == 5) : 
                    cio_action.append(-1)
                elif(i == 6) : 
                    cio_action.append(0)
                elif(i == 7) : 
                    cio_action.append(1)
                elif(i == 8) : 
                    cio_action.append(2)
                elif(i == 9) : 
                    cio_action.append(3)
                elif(i == 10) : 
                    cio_action.append(4)
                elif(i == 11) : 
                    cio_action.append(5)
                else : 
                    cio_action.append(6)            
         
            print("CIO action: ", cio_action)

            next_state, reward, done, _ = env.step(cio_action)
            
            if next_state is None:
                if time != 119 :
                    break_ep = break_ep +1
                    EPISODES = EPISODES+1
                else:
                    FullStep_Ep.append(e+1)
                
                break
            
            print("break_ep: ",break_ep)
            print("Full Step Episode: ",FullStep_Ep)
            
            next_state1 = np.reshape(next_state['rbUtil'], [5,1])
            next_state1 = next_state1[2:5,0:1]

            next_MCSVec = np.array(next_state['MCSPen'])
            next_state2 = np.sum(next_MCSVec[:,:10], axis=1)
            next_state2 = np.reshape(next_state2, [5,1])
            next_state2 = next_state2[2:5,0:1]

            next_state3 = np.reshape(next_state['ServedUes'], [5,1])
            next_state3 = next_state3[2:5,0:1]

            next_state4 = np.reshape(next_state['Throughput'], [5,1])
            next_state4 = next_state4[2:5,0:1]

            R_rewards = np.reshape(next_state['Throughput'], [5,1])
            R_rewards = R_rewards[2:5,0:1]
            RewardSum = R_rewards[0] + R_rewards[1] + R_rewards[2]
            print("RewardSum :",RewardSum)

            R_rewards = [j/100 for sub in R_rewards for j in sub]

            next_state = np.concatenate( (next_state1, next_state2, next_state3, next_state4), axis=None )
            next_state = np.reshape(next_state, [1,state_size])
            
            # Save Results
            # if(time==0):
            #     with open("/home/mnc2/Eunsok/Baseline/NS3_SON/Reward_1172.txt", 'w',encoding="UTF-8") as k:
            #         k.write(str(RewardSum)+"\n")
            #     with open("/home/mnc2/Eunsok/Baseline/NS3_SON/Loss_1172.txt", 'w',encoding="UTF-8") as r:
            #         r.write(str(agent3.loss)+"\n")
            # else:
            #     with open("/home/mnc2/Eunsok/Baseline/NS3_SON/Reward_1172.txt", 'a',encoding="UTF-8") as k:
            #         k.write(str(RewardSum)+"\n")
            #     with open("/home/mnc2/Eunsok/Baseline/NS3_SON/Loss_1172.txt", 'a',encoding="UTF-8") as r:
            #         r.write(str(agent3.loss)+"\n")


            agent3.remember(state, action3, R_rewards[0], next_state)
            agent4.remember(state, action4, R_rewards[1], next_state)
            agent5.remember(state, action5, R_rewards[2], next_state)

            state = next_state

            agent3.learn()
            agent4.learn()
            agent5.learn()

            if((time%24) == 0) :
                 print("Target network update")

                 agent3.update_target_model()
                 agent4.update_target_model()
                 agent5.update_target_model()