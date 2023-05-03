#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from ns3gym import ns3env
# from action_func import*
from collections import OrderedDict
import numpy as np

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2018, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "gawlowicz@tkn.tu-berlin.de"


def action_func(obs,prev_action,numOfenb):
    action = OrderedDict()
    off = prev_action['Offset']
    step = prev_action['Step']

    # Developing a new algorithm for this part
    
    off[0] = 0
    off[1] = 0
    off[2] = 0
    off[3] = 0
    off[4] = 0
    

    action['Offset'] = off
    action['Step'] = step
    
    return action


startSim = False
iterationNum = 120
port = 1150
simTime = 60 # seconds
stepTime = 0.5  # seconds
seed = 0
simArgs = {"--simTime": simTime,
           "--stepTime": stepTime,
           "--testArg": 123}
debug = False

env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim,
                    simSeed=seed, simArgs=simArgs, debug=debug)
env.reset()

ob_space = env.observation_space
ac_space = env.action_space          
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.dtype)

stepIdx = 0
currIt = 0

try:
    while True:
        print("Start iteration: ", currIt)
        obs = env.reset()
        print("Step: ", stepIdx)
        print("---obs: ", obs)
        
        if(stepIdx<2):
            numOfenb = 5

        while True:
            stepIdx += 1

            #New Part
            if(stepIdx==1):
                prev_action = OrderedDict()
                prev_action['Offset'] = []
                prev_action['Step'] = []
                prev_action['Offset'].append(0)
                prev_action['Step'].append(0)
                for i in range(numOfenb-1):
                    prev_action['Offset'].append(0)
                    prev_action['Step'].append(0)
            
            action = action_func(obs,prev_action,numOfenb)
            prev_action = action

            print("---action: ", action['Offset'])

            print("Step: ", stepIdx)
            obs, reward, done, info = env.step(action['Offset'])
            # print("obs : ",obs)
            # print("done : ",done)
            print("---obs, reward, done, info: ", obs, reward, done, info)     

            if done:
                stepIdx = 0
                if currIt + 1 < iterationNum:
                    env.reset()
                break

        currIt += 1
        if currIt == iterationNum:
            break

except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    env.close()
    print("Done")