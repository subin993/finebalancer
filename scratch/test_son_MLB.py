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
    dlPrbUsage = obs['rbUtil']
    off = prev_action['Offset']
    step = prev_action['Step']

    # # Using MLB Algorithm
    # #New Part
    # for i in range(numOfenb):
    #     if off[i] >= -24 and off[i] <= 24:
    #         if dlPrbUsage[i] > 20:
    #             if off[i] <= -6:
    #                 off[i] -= 2
    #             elif off[i] <= 6:
    #                 off[i] -= 1
    #             elif off[i] == -24:
    #                 off[i] -= 0
    #             else:
    #                 off[i] -= 2
    #             if step[i] ==1:
    #                 step[i] = 0
    #             if dlPrbUsage[i] == 100:
    #                 step[i] += 1
    #         if dlPrbUsage[i] <= 20:
    #             step[i] +=1
    #             if step[i]==4:
    #                 if off[i] <= -8:
    #                     off[i] += 2
    #                 elif off[i] < 6:
    #                     off[i] +=1
    #                 elif off[i] == 24:
    #                     off[i] += 0
    #                 else:
    #                     off[i] +=2
    #                 step[i]=0

    # # Non-using MLB Algorithm
    # #New Part
    for i in range(numOfenb):
        if off[i] > -6 and off[i] < 6:
            if dlPrbUsage[i] > 40:
                if off[i] <= -6:
                    off[i] -= 0
                elif off[i] <= 6:
                    off[i] -= 0
                else:
                    off[i] -= 0
                if step[i] ==1:
                    step[i] = 0
                if dlPrbUsage[i] == 100:
                    off[i] -= 1
                    step[i] += 1
            if dlPrbUsage[i] <= 40:
                step[i] +=1
                if step[i]==2:
                    if off[i] <= -8:
                        off[i] += 0
                    elif off[i] < 6:
                        off[i] +=0
                    else:
                        off[i] +=0
                    step[i]=0

    action['Offset'] = off
    action['Step'] = step
    
    return action


startSim = False
iterationNum = 100
port = 1122
simTime = 5 # seconds
stepTime = 1  # seconds
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
            numOfenb = len(obs['rbUtil'])

        while True:
            stepIdx += 1
            
            #Original Part
            #action = OrderedDict()
            #action_list = np.random.randint(-6,7,4)
            #action['Offset'] = action_list

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
            # action = OrderedDict()
            # action['Offset'] = prev_action['Offset']

            print("---action: ", action['Offset'])

            print("Step: ", stepIdx)
            obs, reward, done, info = env.step(action['Offset'])
            # print("obs : ",obs)
            # print("done : ",done)
            print("---obs, reward, done, info: ", obs, reward, done, info)
            dlPrbUsage = obs["rbUtil"]
            
            print("---dlPrbUsage: ", dlPrbUsage)
            

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