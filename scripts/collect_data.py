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

def get_env_params(env):
    ''' Extract the environment parameters '''
    # Action Dimension (for a single agent)
    action_space = env.action_space         # list of agents' action spaces, each is a gym box 
    s_action_dim = action_space[0].shape[0]

    # States (for a single agent)
    state_space  = env.observation_space     # list of agents' state spaces, each is a gym box  
    s_state_dim   = state_space[0].shape[0]

    # Constraints (multiagent)
    constraint_space = env.constraint_space
    s_constraint_dim   = constraint_space[0].shape[0]

    # Number of agents
    num_agents = len(state_space)

    # Return state/action/constraint dimensions from the centralized agent's POV
    state_dim      = s_state_dim * num_agents
    action_dim     = s_action_dim * num_agents
    constraint_dim = s_constraint_dim * num_agents

    #assert num_agents == len(action_space)
    return state_dim, action_dim, constraint_dim, num_agents

def main():

    # Experiment Configuration
    episodes          = 2000
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

    # Parse experiment dimensions
    state_dim, action_dim, constraint_dim, num_agents = get_env_params(env)

    # Data Storage Containers
    size = episodes*(steps_per_episode - 1)
    state_buf       = np.zeros([size, state_dim])
    action_buf      = np.zeros([size, action_dim])
    constraint_diff = np.zeros([size, constraint_dim])

    # Simulate the environment and generate dataset for constraints networks
    for episode in range(episodes):
        print(f'episode={episode}')

        # Episode "Preprocessing"
        state          = env.reset()
        constraint_old = np.zeros([constraint_dim])

        for step in range(steps_per_episode):

            # Simulation
            action = np.random.uniform(-1, 1, action_dim)
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
