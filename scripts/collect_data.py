#!/usr/bin/env python
import operator
import os,sys
import numpy as np
import pandas as pd
import copy
#sys.path.insert(1, os.path.join(sys.path[0], '..'))
#import argparse

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios
np.random.seed(2021)

def main():

    # Experiment Configuration
    episodes          = 5000
    steps_per_episode = 200
    output_dir        = '../data/'

    # Load the simulation scenario
    scenario = scenarios.load("decentralized_safe.py").Scenario()
    world    = scenario.make_world()

    # Environment Setup
    env = MultiAgentEnv(world,
                        scenario.reset_world,
                        scenario.reward,
                        scenario.observation,
                        info_callback=None,
                        constraint_callback = scenario.constraints,
                        shared_viewer = True)

    # The scenario parameters
    env_params = env.get_env_parameters()
    state_dim = env_params["state_dim"]
    action_dim   = env_params["act_dim"]
    constraint_dim = env_params["constraint_dim"]
    num_agents = env_params["num_agents"]
    
    # Data Storage Containers
    size = episodes*(steps_per_episode - 1)
    state_buf       = np.zeros([size, state_dim*num_agents])
    action_buf      = np.zeros([size, action_dim*num_agents])
    constraint_diff = np.zeros([size, constraint_dim*num_agents])

    # Simulate the environment and generate dataset for constraints networks
    for episode in range(episodes):
        print(f'episode={episode}')

        # Episode "Preprocessing"
        state          = env.reset()
        constraint_old = np.zeros([constraint_dim])

        for step in range(steps_per_episode):

            # Simulation
            action = np.random.uniform(-1, 1, action_dim*num_agents)
            action = np.split(action, num_agents)

            # Deep Copy the agent's action (otherwise it's altered in env.step())
            action_copy = copy.deepcopy(action)
            next_state, reward,_ ,_ , constraint = env.step(action_copy)

            # Omit first simulation step
            if step == 0:
                constraint_old = constraint
                continue

            # Constraint diff 
            diff            = list(map(operator.sub, constraint, constraint_old))
            constraint_old = constraint

            # Store stuff to buffers for training
            idx = episode*(steps_per_episode - 1) + step - 1
            state_buf[idx,:]       = np.concatenate(state)
            action_buf[idx,:]      = np.concatenate(action)
            constraint_diff[idx,:] = np.concatenate(diff)

            # update state
            state = next_state

    # Export Results for training
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pd.DataFrame(state_buf).to_csv(output_dir + "D_state_decentralized.csv")
    pd.DataFrame(action_buf).to_csv(output_dir + "D_action_decentralized.csv")
    pd.DataFrame(constraint_diff).to_csv(output_dir + "D_constraint_decentralized.csv")
    print("Done... Data saved")

if __name__ == "__main__":
    main()
