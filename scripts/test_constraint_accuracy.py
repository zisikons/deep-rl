#!/usr/bin/env python
import torch
import operator
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
import pandas as pd

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios

#from core.DDPG import DDPGagent
#from Noise import OUNoise
from operator import add
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from core.constraint_network import ConstraintNetwork

import ipdb
import copy

from sklearn.metrics import mean_squared_error

def get_env_params(env):
    ''' Extract the environment parameters '''
    action_space = env.action_space         # list of agents' action spaces, each is a gym box 
    action_dim = action_space[0].shape[0]

    state_space = env.observation_space     # list of agents' state spaces, each is a gym box  
    state_dim = state_space[0].shape[0]

    constraint_space = env.constraint_space
    constraint_dim = constraint_space[0].shape[0]

    num_agents = len(state_space)

    assert num_agents == len(action_space)
    return state_dim, action_dim, num_agents, constraint_dim

def is_collide(scenario, agents):
    collision1 = scenario.is_collision(agents[0], agents[1])
    collision2 = scenario.is_collision(agents[1],agents[2])
    collision3 = scenario.is_collision(agents[0],agents[2])
    collisions = [collision1, collision2, collision3]
    if any(collisions) == True:
        return True
    else:
        return False

@torch.no_grad()
def main():
    scenario = scenarios.load("centralized_safe.py").Scenario()
    world    = scenario.make_world()

    # simulation properties
    episodes          = 100
    steps_per_episode = 200
    size = episodes*(steps_per_episode - 1)

    # Environment Setup
    '''
    env = MultiAgentEnv(world,
                        scenario.reset_world,
                        scenario.reward,
                        scenario.observation,
                        scenario.constraints,
                        info_callback=None,
                        shared_viewer = True)
    '''

    # Environment Setup
    env = MultiAgentEnv(world,
                        scenario.reset_world,
                        scenario.reward,
                        scenario.observation,
                        info_callback=None,
                        constraint_callback = scenario.constraints,
                        shared_viewer = True)

    # networks directory 
    abs_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    constraint_networks_dir = abs_path + '../data/constraint_networks/'

    # Environment Params
    state_dim, act_dim, num_agents, constraint_dim = get_env_params(env)

    # Centrallized Dimensions
    cent_state_dim = state_dim * num_agents
    cent_act_dim   = act_dim * num_agents

    # Import Constraint Networks
    constraint_network = cent_act_dim* [ConstraintNetwork(cent_state_dim, cent_act_dim)]

    for i in range(len(constraint_network)):
        constraint_network[i].load_state_dict(torch.load(constraint_networks_dir + "constraint_net_"+str(i)+".pkl"))


    # don't remember why this line is here
    agents = env.world.agents

    # Simulate the environment and generate data
    y_pred = np.zeros([size,cent_act_dim])
    y_true = np.zeros([size,cent_act_dim])

    for episode in range(episodes):

        # Episode "Preprocessing"
        state           = env.reset()
        constraints_old = np.zeros([constraint_dim*num_agents]) # useful?

        for step in range(steps_per_episode):

            # Simulation
            action = np.random.uniform(low = -1, high = 1,size= num_agents*act_dim)
            action = np.split(action, num_agents)
            action_copy = copy.deepcopy(action)

            #next_state, reward, done, constraints, *rest = env.step(action_copy)
            next_state, reward,done ,_ , constraints = env.step(action_copy)

            # Omit first simulation step
            if step == 0:
                constraints_old = constraints
                continue

            # Prediction vs reality
            for i in range(cent_act_dim):

                # 0 order approx
                const_term = np.concatenate(constraints_old)[i]

                # 1st order approximation prediction
                torch_state = torch.Tensor(np.concatenate(state)).unsqueeze(0)
                constraint_approx = constraint_network[i](torch_state).numpy() @ np.concatenate(action)

                # Store results
                idx = episode*(steps_per_episode - 1) + step - 1
                y_pred[idx, i] = const_term + constraint_approx
                y_true[idx, i] = np.concatenate(constraints)[i]

                diff = y_pred[idx, i] - y_true[idx, i]


                agent_idx = i // 2
                if i == 15:
                    velocity  = np.linalg.norm(state[agent_idx][0:2])
                    print("Diff = " + str(diff) + "|| Agent's speed = " + str(velocity))
                #if np.abs(y_pred[idx, i] - y_true[idx, i]) > 10:
                #    ipdb.set_trace()

            # update state
            state = next_state
            constraints_old = constraints

    MSE  = np.zeros(cent_act_dim)

    for i in range(cent_act_dim):
        MSE[i] = mean_squared_error(y_true[:,i], y_pred[:, i])

    print(MSE)
    diff = y_pred - y_true
    pd.DataFrame(diff).to_csv("constr_offset.csv")
    ipdb.set_trace()

    return
    pd.DataFrame(y_true).to_csv("constr_true.csv")
    pd.DataFrame(y_pred).to_csv("constr_pred.csv")
    #pd.DataFrame(next_state_buf).to_csv("D_next_state.csv")
    pd.DataFrame(constraints_diff).to_csv("DD_constraints.csv")
    print("Done... Data saved")

if __name__ == "__main__":
    main()
