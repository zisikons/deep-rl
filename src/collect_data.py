#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
import pandas as pd

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios

from DDPG import DDPGagent
from NormalizeEnv import NormalizedEnv
from Noise import OUNoise
from operator import add
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import ipdb

def is_collide(scenario, agents):
    collision1 = scenario.is_collision(agents[0], agents[1])
    collision2 = scenario.is_collision(agents[1],agents[2])
    collision3 = scenario.is_collision(agents[0],agents[2])
    collisions = [collision1, collision2, collision3]
    if any(collisions) == True:
        return True
    else:
        return False

def main():
    scenario = scenarios.load("my_scenario.py").Scenario()
    world = scenario.make_world()
    
    # environment properties
    num_agents = 3
    act_dim = 2
    state_dim = 14

    episodes = 1000
    steps_per_episode = 200
    size = episodes*steps_per_episode 
    
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = True)     
     
    # data storage containers
    state_buf      = np.zeros([size, state_dim*num_agents])
    next_state_buf = np.zeros([size, state_dim*num_agents]) 
    actions_buf    = np.zeros([size, act_dim*num_agents])
    
    # Initialize the environment
    state, ep_ret, ep_len = env.reset(), 0, 0
    
    agents = env.world.agents

    for episode in range(episodes):  
        state = env.reset()
        for step in range(steps_per_episode):

            action = np.random.uniform(low = -1, high = 1,size= num_agents*act_dim)
            action = np.split(action, num_agents)
            next_state, reward, done, *rest = env.step(action)
            
            state_buf[episode*steps_per_episode + step,:]      = np.concatenate(state)
            actions_buf[episode*steps_per_episode + step,:]    = np.concatenate(action)
            next_state_buf[episode*steps_per_episode + step,:] = np.concatenate(next_state) 

            # update state
            state = next_state
            
            if all(done) == True or is_collide(scenario, agents):
                state = env.reset()

    print("Done... Data saved")
    pd.DataFrame(state_buf).to_csv("D_state.csv")
    pd.DataFrame(actions_buf).to_csv("D_action.csv")
    pd.DataFrame(next_state_buf).to_csv("D_next_state.csv") 

             
if __name__ == "__main__":
    main()
