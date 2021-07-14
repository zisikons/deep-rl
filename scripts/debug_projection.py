#!/usr/bin/env python
import os,sys
import copy
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
import torch
from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios

from core.SafeMADDPG import SafeMADDPGagent
from core.Noise import OUNoise

from gym.wrappers.monitoring.video_recorder import VideoRecorder
import matplotlib.pyplot as plt
import ipdb

def check_collision(state):

    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            dist = np.linalg.norm(state[i][2:4] - state[j][2:4])
            print(f'Agents ({i} , {j}) = {dist}')
            if np.linalg.norm(state[i][2:4] - state[j][2:4]) < 0.3:
                print(f'Agent {i} collides with agent {j}')
                return



def main():


    # Usefull Directories
    abs_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    constraint_networks_dir = abs_path + '../data/constraint_networks_MADDPG/'

    # Load the simulation scenario
    scenario = scenarios.load("decentralized_safe.py").Scenario()
    world    = scenario.make_world()

    # Environment Setup
    env = MultiAgentEnv(world,
                        scenario.reset_world,
                        scenario.reward,
                        scenario.observation,
                        info_callback=None,
                        done_callback = scenario.done,
                        constraint_callback = scenario.constraints,
                        shared_viewer = True)

    # get the scenario parameters
    env_params = env.get_env_parameters()
    state_dim = env_params["state_dim"]
    act_dim   = env_params["act_dim"]
    constraint_dim = env_params["constraint_dim"]
    N_agents = env_params["num_agents"]
    print(env_params)

    # Training Parameters
    batch_size = 128
    episodes = 8000
    steps_per_episode = 300
    agent_update_rate = 100


    # MADDPG Agent
    agent = SafeMADDPGagent(state_dim = state_dim,
                            act_dim = act_dim,
                            N_agents = N_agents,
                            batch_size = batch_size,
                            constraint_dim = constraint_dim,
                            constraint_networks_dir=constraint_networks_dir,
                            soften = False)

    # Load problematic data
    data = np.load('problem.npz')
    state      = data['arr_0']
    action     = data['arr_1']
    constraint = data['arr_2']

    # Call projection code
    corrected_action = agent.correct_actions_hard2(state, action, constraint)

    # Evaluate networks
    #actions = np.concatenate(action)
    state = torch.tensor(np.concatenate(state))

    # (1) Problem Variables
    # Problem specific constants
    #I    = np.eye(self.total_action_dim)
    #ones = np.ones(self.total_action_dim)
    #C    = np.concatenate(constraint)

    # Formulate the constraints using neural networks
    G    = np.zeros([agent.total_action_dim, agent.total_action_dim])
    for i, net in enumerate(agent.constraint_nets):
        G[i, :] = net(state).detach().numpy()


    predicted_constr = constraint.flatten() + G @ corrected_action
    ipdb.set_trace()
    # Specify agents to plot
    i = 0
    j = 2

    # Get related data
    pos_i = state[i][2:4]
    pos_j = state[j][2:4]

    vel_i = state[i][0:2]
    vel_j = state[j][0:2]

    scale_factor = 0.1
    original_action_i = scale_factor * action[i]
    original_action_j = scale_factor * action[j]

    corrected_action_i = scale_factor * corrected_action.reshape(3,2)[i]
    corrected_action_j = scale_factor * corrected_action.reshape(3,2)[j]

    plt.figure()
    # Plot positions
    plt.scatter(pos_i[0], pos_i[1], s=600, alpha=0.5)
    plt.scatter(pos_j[0], pos_j[1], s=600, alpha=0.5)
    #plt.show()

    # Plot velocities
    plt.arrow(pos_i[0], pos_i[1], vel_i[0], vel_i[1])
    plt.arrow(pos_j[0], pos_j[1], vel_j[0], vel_j[1])

    # Plot original action
    plt.arrow(pos_i[0], pos_i[1], original_action_i[0], original_action_i[1], color='green')
    plt.arrow(pos_j[0], pos_j[1], original_action_j[0], original_action_j[1], color='green')

    # Plot original action
    plt.arrow(pos_i[0], pos_i[1], corrected_action_i[0], corrected_action_i[1], color='red')
    plt.arrow(pos_j[0], pos_j[1], corrected_action_j[0], corrected_action_j[1], color='red')


    plt.show()


if __name__ == "__main__":
    main()
