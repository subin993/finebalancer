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

# GPU 사용가능 -> True , GPU 사용 불가능 -> False
# print(torch.cuda.is_available())

# # GPU 사용가능 -> 가장 빠른 GPU 사용, GPU 사용 불가능 -> CPU 자동 지정
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # GPU 이름 체크
# print(torch.cuda.get_device_name(), device = 0)

# # 사용 가능한 GPU 개수 체크
# print(torch.cuda.device_count())

EPISODES = 5 # Original value is 200
max_env_steps = 1000  # Maximum number of steps in every episode (origianl value is 50)
port=1131 # Should be consistent with NS-3 simulation port
stepTime=0.2
startSim=0
seed=3
simArgs = {}
debug=True
# cio_action = [0,0,0,0,0] # 처음 CIO Action
cio_action = [0,0,0] # 처음 CIO Action
action_prev = [] # 이전 action을 저장하기 위함

# DQN Agent Class
################################################################
class DQNAgent:
    def __init__(self, state_size, action_size):
        # Input / Ouput size
        self.state_size = state_size
        self.action_size = action_size
        # Replay Memory
        self.memory = deque(maxlen=20000)
        # Hyperparameter
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1
        self.epsilon_end = 0.01
        self.epsilon_decay = 200 # 500 step일 때 epsilon = 0.09, epsilon_decay가 100이면 500 step일 때 epsilon = 0.02
        self.steps_done = 0
        self.learning_rate = 0.001
        # Creating main network & target network
        self.model = self._build_model()
        self.target_model = self._build_model()
        # Creating optimizer
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        # Copy main network to target network
        self.update_target_model()

    # Creating Model (GPU)
    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24,24),
            nn.ReLU(),
            nn.Linear(24,self.action_size)
        )
        return model.cuda()
    
    def update_target_model(self):
        # copy weights from the CIO selection network to target network
        self.target_model.load_state_dict(self.model.state_dict())
    
    # Stroing sample in replay memory
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, torch.FloatTensor([reward]).cuda(), torch.FloatTensor([next_state]).cuda()))
    
    # Selecting action by epsilon-greedy method
    def act(self, state):
        # Epsilon-greedy
        # 이미 학습된 모델로 진행될 때는 epsilon-greedy 부분을 주석처리해버리면 될듯
        eps_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        
        act_values = self.model(state)

        if np.random.rand() <= eps_threshold:
            return torch.LongTensor([random.randrange(self.action_size)]).cuda() # [[0]] , [[1]] , [[2]]

        return torch.argmax(act_values[0]).unsqueeze(0)  # returns action index (0 : CIO -1 , 1 : CIO 0 , 2 : CIO +1)
    
    def learn(self):
        
        # 학습을 시작하는 시점을 결정하는 부분
        if len(self.memory) < 2 * batch_size: # 이후에 수정 필요한 부분
            return

        batch = random.sample(self.memory, batch_size) 
        states, actions, rewards, next_states = zip(*batch) 

        # 데이터들의 형태를 맞춰주는 부분

        # states의 형태
        #############################################################
        # tensor([[eNB 별 PRB Usage 5개],
        #         [eNB 별 PRB Usage 5개],
        #         ...batch_size 수만큼 sample이 존재])
        #############################################################
        states = torch.cat(states).squeeze() 
        # print("states : ",states)
        
        # actions의 형태
        #############################################################
        # tensor([[DQN이 취했던 action의 index 1개],
        #         [DQN이 취했던 action의 index 1개],
        #         ...batch_size 수만큼 sample이 존재])
        #############################################################
        actions = torch.cat(actions).unsqueeze(1)
        # print("actions : ",actions)

        # rewards의 형태
        #############################################################
        # tensor([[해당 eNB에 속한 UE들의 평균 throughput(Reward) 1개],
        #         [해당 eNB에 속한 UE들의 평균 throughput(Reward) 1개],
        #         ...batch_size 수만큼 sample이 존재])
        #############################################################
        rewards = torch.cat(rewards).unsqueeze(1)
        # print("rewards : ",rewards)
        
        # next_state의 형태
        #############################################################
        # tensor([[eNB 별 PRB Usage 5개],
        #         [eNB 별 PRB Usage 5개],
        #         ...batch_size 수만큼 sample이 존재])
        #############################################################
        next_states = torch.cat(next_states).squeeze()
        # print("next_states : ",next_states)

        # 학습에 필요한 action value / target value 계산하는 부분
        # current_q, max_next_q, expected_q의 형태는 actions의 형태와 동일 -> Agent 수(5)x1 형태
        current_q = self.model(states).gather(1,actions) # action value
        max_next_q = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        expected_q = rewards + (self.gamma * max_next_q) # target value
        # print("current q : ",current_q)
        # print("max_next q : ",max_next_q)
        # print("expected q : ",expected_q)

        # Loss를 구하고 optimizer로 학습하는 부분
        loss = F.mse_loss(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
################################################################

# Main Code
################################################################
if __name__ == "__main__" :
    # C++ 환경 불러오고 input/output 크기 설정
    env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
    env._max_epsiode_steps = max_env_steps
    ac_space = env.action_space # Getting the action space
    state_size = 3 # Num of observation
    action_size = 3

    # 각 eNB 별 Agent 생성
    agent1 = DQNAgent(state_size, action_size)
    agent2 = DQNAgent(state_size, action_size)
    agent3 = DQNAgent(state_size, action_size)
    # agent4 = DQNAgent(state_size, action_size)
    # agent5 = DQNAgent(state_size, action_size)

    # 이미 저장되있는 모델의 가중치를 불러오는 부분
    # agent1.model.load_state_dict(torch.load('eNB1_Model_test_eunsok.pt'))
    # agent1.target_model.load_state_dict(torch.load('eNB1_Model_target_test_eunsok.pt'))
    # agent2.model.load_state_dict(torch.load('eNB2_Model_test_eunsok.pt'))
    # agent2.target_model.load_state_dict(torch.load('eNB2_Model_target_test_eunsok.pt'))
    # agent3.model.load_state_dict(torch.load('eNB3_Model_test_eunsok.pt'))
    # agent3.target_model.load_state_dict(torch.load('eNB3_Model_target_test_eunsok.pt'))

    done = False
    # 학습을 진행할 sample 단위 : 32
    batch_size = 32
    # reward_sum = []

    for e in range(EPISODES):
        # 처음에 environment로 부터 state (PRB Usage)와 reward (throughput) 받아와서 형변환 하는 부분
        state = env.reset()
        state1 = np.reshape(state['rbUtil'], [3,1])
        R_rewards = np.reshape(state['rewards'], [3,1])
        # state1 = np.reshape(state['rbUtil'], [5,1])
        # R_rewards = np.reshape(state['rewards'], [5,1])
        R_rewards = [j for sub in R_rewards for j in sub]

        state = state1
        state = np.reshape(state, [1,state_size])
   
        for time in range(max_env_steps): # 50 step = 1 episode
            print("*******************************")
            print("episode: {}/{}, step: {}".format(e+1, EPISODES, time))

            # C++로부터 받아온 state는 텐서 형태가 아니므로 텐서 형태 (GPU)로 변환
            state = torch.FloatTensor([state]).cuda()

            # Changed part (Model을 처음 학습할 때 사용)
            # 500 step 까지는 Rule-based로 진행하면서 학습하고 이후는 학습된 모델로 진행
            # Rule-based의 기준은 0.7 이상 정도면 과부하된 것을 판단, 0.2 보다 작으면 해당 기지국의 CIO를 높인다
            if (time <500):
                action_rule = []
                state_rule = state.squeeze().squeeze().tolist()
                for i in range(3) :
                    if(state_rule[i] > 0.9) : action_rule.append(int(0))
                    elif(state_rule[i] < 0.76) : action_rule.append(int(2))
                    else : action_rule.append(int(1))
                action1 = torch.LongTensor([action_rule[0]]).cuda()
                action2 = torch.LongTensor([action_rule[1]]).cuda()
                action3 = torch.LongTensor([action_rule[2]]).cuda()
            else :
                # 500 step 이후로는 입실론 그리디로 진행
                action1 = agent1.act(state)
                action2 = agent2.act(state)
                action3 = agent3.act(state)


            # Original part (이미 학습시킨 Model이 있을 때 사용)
            # Agent 별로 취할 action 결정
            # action1 = agent1.act(state)
            # action2 = agent2.act(state)
            # action3 = agent3.act(state)
            # action4 = agent4.act(state)
            # action5 = agent5.act(state)

            ##
            print('action1 : ',action1.item())
            print('action2 : ',action2.item())
            print('action3 : ',action3.item())
            # print('action4 : ',action4.item())
            # print('action5 : ',action5.item())

            # 현재 결정한 action은 텐서 형태로 c++에 보낼 수 없으므로 일반적인 int 형태로 형변환
            action = []
            action.append(action1.item())
            action.append(action2.item())
            action.append(action3.item())
            # action.append(action4.item())
            # action.append(action5.item())

            
            # 결정한 action을 environment에 취하는 부분
            # -6~6 사이에서만 CIO가 변화하도록 만들고 범위를 넘게 하는 액션은 CIO 조정 0 액션으로 변경

            # Changed part
            # CIO 조정을 너무 자주하는 것 같아 같은 액션을 2번 연속할 때만 CIO 조정을 하도록 변경하였다 (Action의 종류가 많아진다면 이 부분은 없어도 될듯)
            if(time == 0) : action_prev = action
    
            idx = 0
            for i in action :
                if(i == 0) : 
                    if(action_prev[idx] == 0) :
                        if(cio_action[idx]>-6) : 
                            cio_action[idx] -= 1
                        else :
                            action[idx] = 1
                elif(i == 2) : 
                    if(action_prev[idx] == 2) :
                        if(cio_action[idx]<6) :
                            cio_action[idx] += 1
                        else : 
                            action[idx] = 1
                idx += 1
            
            action_prev = action

            # Original part
            # idx = 0
            # for i in action :
            #     if(i == 0) : 
            #         if(cio_action[idx]>-6) : 
            #             cio_action[idx] -= 1
            #         else :
            #             action[idx] = 1
            #     elif(i == 2) : 
            #         if(cio_action[idx]<6) :
            #             cio_action[idx] += 1
            #         else : 
            #             action[idx] = 1
            #     idx += 1
            
            ##
            print("CIO action: ", cio_action)

            # Environment에 action을 취해서 next_state와 reward를 받는 부분
            next_state, reward, done, _ = env.step(cio_action)
            
            # 의도치 않게 환경에서 샘플을 받아오지 못했을 때를 고려한 부분인데 제대로 작동하는 것 같지는 않다
            if next_state is None:
                EPISODES = EPISODES+1
                break
            
            # Environment로 부터 받아온 next_state와 reward 형변환
            state1 = np.reshape(next_state['rbUtil'], [1,3])
            R_rewards = np.reshape(next_state['rewards'], [3,1])
            # state1 = np.reshape(next_state['rbUtil'], [1,5])
            # R_rewards = np.reshape(next_state['rewards'], [5,1])

            # Reward의 값이 너무 커서 target value가 무시되는 현상을 해결하기 위해 Reward에 100을 나눠주었음
            R_rewards = [j/100 for sub in R_rewards for j in sub]
            
            # 출력을 위한 부분
            index = 1
            for i in R_rewards :
                print("eNB {}'s average Throughput : {} ".format(index,i))
                index += 1
            
            next_state = state1

            # Storing sample in replay memory
            agent1.remember(state, action1, R_rewards[0], next_state)
            agent2.remember(state, action2, R_rewards[1], next_state)
            agent3.remember(state, action3, R_rewards[2], next_state)
            # agent4.remember(state, action4, R_rewards[3], next_state)
            # agent5.remember(state, action5, R_rewards[4], next_state)

            state = next_state

            # Learning
            agent1.learn()
            agent2.learn()
            agent3.learn()
            # agent4.learn()
            # agent5.learn()

            # Copy main network to target network
            if((time%24) == 0) :
                ## 
                print("Target network update")

                agent1.update_target_model()
                agent2.update_target_model()
                agent3.update_target_model()
                # agent4.update_target_model()
                # agent5.update_target_model()
            
            if ((time % 200 == 1)&(time != 1)) :
                # Save Model
                print("Save each model")
                torch.save(agent1.model.state_dict(),'eNB1_Model_test_eunsok.pt')
                torch.save(agent2.model.state_dict(),'eNB2_Model_test_eunsok.pt')
                torch.save(agent3.model.state_dict(),'eNB3_Model_test_eunsok.pt')
                torch.save(agent1.target_model.state_dict(),'eNB1_Model_target_test_eunsok.pt')
                torch.save(agent2.target_model.state_dict(),'eNB2_Model_target_test_eunsok.pt')
                torch.save(agent3.target_model.state_dict(),'eNB3_Model_target_test_eunsok.pt')
        
################################################################
                