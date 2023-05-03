import random
import math 
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from ns3gym import ns3env
import matplotlib.pyplot as plt

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		
		self.discount = discount
		self.tau = tau

		self.actor_loss = 0
		self.critic_loss = 0
		
		self.policy_noise = policy_noise * max_action
		self.noise_clip = noise_clip * max_action
		
		self.policy_freq = policy_freq

		# Number of times the learning has been conducted
		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)

		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=32):
		self.total_it += 1

		state, action, next_state, reward = replay_buffer.sample(batch_size)

		with torch.no_grad():
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action) 

			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + self.discount * target_Q

		current_Q1, current_Q2 = self.critic(state, action)

		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		print("Loss: ",critic_loss)

		if self.total_it % self.policy_freq == 0:

			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

			self.actor_loss = actor_loss
			self.critic_loss = critic_loss
			
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)

# Replay Buffer
class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward):

		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
		)

# Main
EPISODES = 10000
max_env_steps = 120 # Maximum number of steps in every episode
port=1150
stepTime=0.5
startSim=0
seed=3
simArgs = {}
debug=True

if __name__ == "__main__" :
    env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
    
    env._max_epsiode_steps = max_env_steps
    
    ac_space = env.action_space 
    max_action = float(6.4999)
    print("max_action: ",max_action)
    
    # Number of states, actions
    state_size = 12 
    action_size = 3 
    
    # Number of UEs
    ueNum = 40

    batch_size = 256

    # Std of Gaussian exploration noise
    expl_noise = 0.1

    start_step = 500

    total_step = 0

    break_ep = 0

    FullStep_Ep = []
    
    agent = TD3(state_size,action_size,max_action)

    replay_buffer = ReplayBuffer(state_size, action_size)

    done = False

    for e in range(EPISODES):
        state = env.reset()

        state1 = np.reshape(state['rbUtil'], [5,1])
        ##
        state1 = state1[2:5,0:1]
        print("rbUtil : ",state1)


        state2 = np.reshape(state['Throughput'], [5,1])
        ##
        state2 = state2[2:5,0:1]
        print("dlThroughput : ",state2)

        MCSVec = np.array(state['MCSPen'])
        state3 = np.sum(MCSVec[:,:10], axis=1)
        state3 = np.reshape(state3, [5,1])
        state3 = state3[2:5,0:1]
        print("MCSPen : ",state3)

        state4 = np.reshape(state['ServedUes'], [5,1])
        state4 = state4[2:5,0:1]
        print("ServedUes : ",state4)

        R_rewards = np.reshape(state['Throughput'], [5,1])
        R_rewards = np.round(R_rewards, 2)
        R_rewards = R_rewards[2:5]
        
        R_rewards = [j for sub in R_rewards for j in sub]
        avg_reward = 0
        for i in range(3):
            avg_reward += R_rewards[i]
        avg_reward = (avg_reward / 3)      
        print("Reward : ",avg_reward)
        
        state = np.concatenate( (state1, state2, state3, state4), axis=None )
        state = np.reshape(state, [1,state_size])

        for time in range(max_env_steps): 
            print("*******************************")
            print("episode: {}/{}, step: {}".format(e+1, EPISODES, time))
            print("Total step: ", total_step)

            if (total_step < start_step):
                print("Random Action")
                action = env.action_space.sample()

                action = action[2:5]
                action = (np.round(action,0)).astype(float)
                env_action = action.tolist()
                env_action.insert(0,0.0)
                env_action.insert(0,0.0)

            else:
                print("Agent Action")
                action = (
					agent.select_action(np.array(state))
					+ np.random.normal(0, max_action * expl_noise, size=action_size)
				).clip(-max_action, max_action)
                
                env_action = (np.round(action,0)).astype(float)

                env_action = env_action.tolist()
                env_action.insert(0,0.0)
                env_action.insert(0,0.0)
			
            for i in range(5):
                env_action.append(0.0)
            
            print("actions: ",action)
            print("env actions: ",env_action)

            next_state, reward, done, _ = env.step(env_action)
            
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
            print("rbUtil : ",next_state1)

            next_state2 = np.reshape(next_state['Throughput'], [5,1])
            next_state2 = next_state2[2:5,0:1]
            print("dlThroughput : ",next_state2)

            next_MCSVec = np.array(next_state['MCSPen'])
            next_state3 = np.sum(next_MCSVec[:,:10], axis=1)
            next_state3 = np.reshape(next_state3, [5,1])
            next_state3 = next_state3[2:5,0:1]
            print("MCSPen : ",next_state3)

            next_state4 = np.reshape(next_state['ServedUes'], [5,1])
            next_state4 = next_state4[2:5,0:1]
            print("ServedUes : ",next_state4)

            R_rewards = np.reshape(next_state['Throughput'], [5,1])
            R_rewards = np.round(R_rewards, 2)
            
            R_rewards = [j for sub in R_rewards for j in sub]
            R_rewards = R_rewards[2:5]

            avg_reward = 0 
            for i in range(3):
                avg_reward += R_rewards[i]
            
            avg_reward = (avg_reward / 3)
            print("Reward : ",avg_reward)

            RewardSum = R_rewards[0] + R_rewards[1] + R_rewards[2]

            # Save Results
            # if(time == 0):
            #     with open("/home/mnc2/Eunsok/Baseline/NS3_SON/Reward_1150.txt", 'w',encoding="UTF-8") as k:
            #         k.write(str(RewardSum)+"\n")
            #     with open("/home/mnc2/Eunsok/Baseline/NS3_SON/Loss_1150.txt", 'w',encoding="UTF-8") as j:
            #         j.write(str(agent.critic_loss)+"\n")
            # else:
            #     with open("/home/mnc2/Eunsok/Baseline/NS3_SON/Reward_1150.txt", 'a',encoding="UTF-8") as k:
            #         k.write(str(RewardSum)+"\n")
            #     with open("/home/mnc2/Eunsok/Baseline/NS3_SON/Loss_1150.txt", 'a',encoding="UTF-8") as j:
            #         j.write(str(agent.critic_loss)+"\n")

            next_state = np.concatenate( (next_state1, next_state2, next_state3, next_state4), axis=None )
            next_state = np.reshape(next_state, [1,state_size])

            if(time != 0):
                replay_buffer.add(state, action, next_state, avg_reward)
                total_step = total_step + 1
            

            state = next_state

            if (total_step > start_step):
                agent.train(replay_buffer, batch_size)