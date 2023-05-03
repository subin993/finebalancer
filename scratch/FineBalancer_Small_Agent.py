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


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		return q1

class MADDPG(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		agent_id, 
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
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

		self.loss = 0
		
		self.policy_noise = policy_noise * max_action
		self.noise_clip = noise_clip * max_action
		
		self.agent_id = agent_id


	
	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)

		return self.actor(state).cpu().data.numpy().flatten()

	
	def train(self, transitions, other_agents, n_agents, batch_size):

		r = transitions['r_%d' % self.agent_id]
		

		s, a, s_n = [], [], []  
		s_main = transitions['s_%d' % self.agent_id]

		for agent_id in range(n_agents):
			s.append(transitions['s_%d' % agent_id])
			a.append(transitions['a_%d' % agent_id])
			s_n.append(transitions['s_n_%d' % agent_id])

		a_n = []
		with torch.no_grad():
			index = 0
			for agent_id in range(n_agents):
				if agent_id == self.agent_id:
					a_n.append(self.actor_target(s_n[agent_id]))
				else:
					a_n.append(other_agents[index].actor_target(s_n[agent_id]))
					index += 1

			s_temp = s[0]
			a_temp = a[0]
			s_n_temp = s_n[0]
			a_n_temp = a_n[0]
			r_temp = r
			for i in range(n_agents-1):
				s_temp = torch.cat((s_temp,s[i+1]),0)
				a_temp = torch.cat((a_temp,a[i+1]),0)
				s_n_temp = torch.cat((s_n_temp,s_n[i+1]),0)
				a_n_temp = torch.cat((a_n_temp,a_n[i+1]),0)
				r_temp = torch.cat((r_temp,r),0)
			
			s = s_temp
			a = a_temp
			s_n = s_n_temp
			a_n = a_n_temp
			r = r_temp
			
			q_next = self.critic_target(s_n, a_n).detach()

			target_q = (r + self.discount * q_next).detach()

		q_value = self.critic(s, a)
		critic_loss = F.mse_loss(q_value, target_q)

		a_actor = self.actor(s_main)
		a[self.agent_id*2] = a_actor[0]
		a[self.agent_id*2+1] = a_actor[1]

		actor_loss = - self.critic(s, a).mean()

		self.loss = actor_loss

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

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
	def __init__(self, state_dim, action_dim, agent_id, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.agent_id = agent_id

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


	def sample(self, batch_size, ind):
		
		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
		)

# Main
EPISODES = 10000 
max_env_steps = 40 # Maximum number of steps in every episode
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
    
    # Number of agents
    n_agents = 3
    
    # Number of states, actions
    state_size = 4
    action_size = 2

    # Number of UEs
    ueNum = 40

    batch_size = 256

    # Epsilon
    epsilon = 1.0
    epsilon_end = 0.000000000000001
    epsilon_decay = 0.9997 

    # Std of Gaussian exploration noise
    expl_noise = 0.1

    break_ep = 0

    agents = []
    replay_buffers = []
    for i in range(n_agents):
        agent = MADDPG(state_size,action_size,max_action,i)
        replay_buffer = ReplayBuffer(state_size, action_size,i)
        
        agents.append(agent)
        replay_buffers.append(replay_buffer)

    done = False

    for e in range(EPISODES):

        state = env.reset()

        state1 = np.reshape(state['AvgCqi'], [5,1])
        ##
        state1 = state1[2:5,0:1]

        state1 = np.round(state1, 2)
        print("AvgCqi : ",state1)

        state1_1 = state1[0:1,0:1]
        state1_2 = state1[1:2,0:1]
        state1_3 = state1[2:3,0:1]

        state2 = np.reshape(state['Throughput'], [5,1])
        ##
        state2 = state2[2:5,0:1]

        state2 = np.round(state2, 2)
        print("Throughput : ",state2)

        state2_1 = state2[0:1,0:1]
        state2_2 = state2[1:2,0:1]
        state2_3 = state2[2:3,0:1]

        state3 = np.reshape(state['FarUes'], [5,1])
        ##
        state3 = state3[2:5,0:1]

        state3 = np.round(state3, 2)
        print("FarUes : ",state3)

        state3_1 = state3[0:1,0:1]
        state3_2 = state3[1:2,0:1]
        state3_3 = state3[2:3,0:1]    

        state4 = np.reshape(state['ServedUes'], [5,1])
        state4 = state4[2:5,0:1]
        print("ServedUes : ",state4)

        state4_1 = state4[0:1,0:1]
        state4_2 = state4[1:2,0:1]
        state4_3 = state4[2:3,0:1]     

        R_rewards = np.reshape(state['Throughput'], [5,1])
        R_rewards = np.round(R_rewards, 2)
        
        R_rewards = [j for sub in R_rewards for j in sub]

        Reward = R_rewards [2:5]
        print("Reward : ",Reward)
        
        eNB1_state = np.concatenate( (state1_1, state2_1, state3_1,state4_1), axis=None )
        eNB2_state = np.concatenate( (state1_2, state2_2, state3_2,state4_2), axis=None )
        eNB3_state = np.concatenate( (state1_3, state2_3, state3_3,state4_3), axis=None )
        eNB1_state = np.reshape(eNB1_state, [1,state_size])
        eNB2_state = np.reshape(eNB2_state, [1,state_size])
        eNB3_state = np.reshape(eNB3_state, [1,state_size])
        
        states = []
        states.append(eNB1_state)
        states.append(eNB2_state)
        states.append(eNB3_state)
        
        for time in range(max_env_steps): 
            print("*******************************")
            print("episode: {}/{}, step: {}".format(e+1, EPISODES, time))

            actions = []
            env_actions = []
            for agent_id, agent in enumerate(agents):
                if (np.random.rand() <= epsilon) :
                    print("Random Action")
                    
                    action1 = random.uniform(-6,6)
                    action2 = random.uniform(-6,6)
                    
                    env_action1 = round(action1 * 0.8,0)
                    env_action2 = round(action2,4)

                    env_action2 = env_action2 * 0.6

                    action = np.concatenate( (action1, action2), axis=None)
                    actions.append(action)

                    env_actions.insert(agent_id,env_action1)
                    env_actions.insert(agent_id+3,env_action2)

                else:
                    print("Agent Action")
                    action = (
					agent.select_action(np.array(states[agent_id]))
					+ np.random.normal(0, max_action * expl_noise, size=action_size)
				    ).clip(-max_action, max_action)

                    actions.append(action)
                    
                    env_action1 = float(np.round(action[0] * 0.8,0).astype(float))
                    env_action2 = float(np.round(action[1],4).astype(float))
                    
                    env_action2 = env_action2 * 0.6

                    env_actions.insert(agent_id,env_action1)
                    env_actions.insert(agent_id+3,env_action2)
                    
            env_actions.insert(0,0.0)
            env_actions.insert(0,0.0)

            env_actions.insert(5,0.0)
            env_actions.insert(5,0.0)

            print("actions: ",actions)
            print("env actions: ",env_actions)

            next_state, reward, done, _ = env.step(env_actions)
            
            if next_state is None:

                if time != 199 :
                    break_ep = break_ep +1
                    EPISODES = EPISODES+1
                
                break

            print("break_ep: ",break_ep)
            
            next_state1 = np.reshape(next_state['AvgCqi'], [5,1])
            next_state1 = next_state1[2:5,0:1]
            
            next_state1 = np.round(next_state1, 2)
            print("AvgCqi : ",next_state1)

            next_state1_1 = next_state1[0:1,0:1]
            next_state1_2 = next_state1[1:2,0:1]
            next_state1_3 = next_state1[2:3,0:1]

            next_state2 = np.reshape(next_state['Throughput'], [5,1])
            next_state2 = next_state2[2:5,0:1]

            next_state2 = np.round(next_state2, 2)
            print("Throughput : ",next_state2)

            next_state2_1 = next_state2[0:1,0:1]
            next_state2_2 = next_state2[1:2,0:1]
            next_state2_3 = next_state2[2:3,0:1]

            next_state3 = np.reshape(next_state['FarUes'], [5,1])
            next_state3 = next_state3[2:5,0:1]

            next_state3 = np.round(next_state3, 2)
            print("FarUes : ",next_state3)

            next_state3_1 = next_state3[0:1,0:1]
            next_state3_2 = next_state3[1:2,0:1]
            next_state3_3 = next_state3[2:3,0:1]

            next_state4 = np.reshape(next_state['ServedUes'], [5,1])
            next_state4 = next_state4[2:5,0:1]
            print("ServedUes : ",next_state4)

            next_state4_1 = next_state4[0:1,0:1]
            next_state4_2 = next_state4[1:2,0:1]
            next_state4_3 = next_state4[2:3,0:1]

            R_rewards = np.reshape(next_state['Throughput'], [5,1])
            R_rewards = np.round(R_rewards, 2)
            R_rewards = [j for sub in R_rewards for j in sub]
            R_rewards = R_rewards[2:5]
            print("Reward : ",R_rewards)

            RewardSum = R_rewards[0] + R_rewards[1] + R_rewards[2]
            
            # Save Results
            # if(time == 0):
            #     with open("/home/mnc/ns3-son/Baseline/NS3_SON/Reward_1162.txt", 'w',encoding="UTF-8") as k:
            #         k.write(str(RewardSum)+"\n")
            # else:
            #     with open("/home/mnc/ns3-son/Baseline/NS3_SON/Reward_1162.txt", 'a',encoding="UTF-8") as k:
            #         k.write(str(RewardSum)+"\n")

            eNB1_next_state = np.concatenate( (next_state1_1, next_state2_1, next_state3_1, next_state4_1), axis=None )
            eNB2_next_state = np.concatenate( (next_state1_2, next_state2_2, next_state3_2, next_state4_2), axis=None )
            eNB3_next_state = np.concatenate( (next_state1_3, next_state2_3, next_state3_3, next_state4_3), axis=None )
            eNB1_next_state = np.reshape(eNB1_next_state, [1,state_size])
            eNB2_next_state = np.reshape(eNB2_next_state, [1,state_size])
            eNB3_next_state = np.reshape(eNB3_next_state, [1,state_size])

            if(time != 0) :
                replay_buffers[0].add(eNB1_state, actions[0], eNB1_next_state, R_rewards[0])
                replay_buffers[1].add(eNB2_state, actions[1], eNB2_next_state, R_rewards[1])
                replay_buffers[2].add(eNB3_state, actions[2], eNB3_next_state, R_rewards[2])

            if (epsilon > epsilon_end) :
                epsilon *= epsilon_decay
            print("Epsilon: ",epsilon)


            eNB1_state = eNB1_next_state
            eNB2_state = eNB2_next_state
            eNB3_state = eNB3_next_state

            states = []
            states.append(eNB1_state)
            states.append(eNB2_state)
            states.append(eNB3_state)

            print("Replay Buffer Size: ", replay_buffers[0].size)

            if replay_buffers[0].size >= batch_size:
                ind = np.random.randint(0, replay_buffers[0].size, size=batch_size)
                transitions = {}
                for i in range(3):
                    state, action, next_state, reward = replay_buffers[i].sample(batch_size, ind)
                    transitions["s_%d" %i] = state
                    transitions["a_%d" %i] = action
                    transitions["r_%d" %i] = reward
                    transitions["s_n_%d" %i] = next_state
                
                for agent in agents:
                    print("Agent Id: ",agent.agent_id, " Loss: ", agent.loss)
                    other_agents = agents.copy()
                    other_agents.remove(agent)
                    agent.train(transitions, other_agents, n_agents, batch_size)