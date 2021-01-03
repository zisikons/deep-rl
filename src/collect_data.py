#!/usr/bin/env python
import operator
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
import pandas as pd

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios

from DDPG import DDPGagent
from Noise import OUNoise
from operator import add
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import ipdb
import copy

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

def main():
    scenario = scenarios.load("my_scenario.py").Scenario()
    world = scenario.make_world()

    # simulation properties
    episodes = 5000
    steps_per_episode = 200
    size = episodes*(steps_per_episode - 1)

    # Environment Setup
    env = MultiAgentEnv(world,
                        scenario.reset_world,
                        scenario.reward,
                        scenario.observation,
                        scenario.constraints,
                        info_callback=None,
                        shared_viewer = True)

    state_dim, act_dim, num_agents, constraint_dim = get_env_params(env)

    # data storage containers
    state_buf        = np.zeros([size, state_dim*num_agents])
    actions_buf      = np.zeros([size, act_dim*num_agents])
    constraints_diff = np.zeros([size, constraint_dim*num_agents])

    agents = env.world.agents

    # Simulate the environment and generate data
    for episode in range(episodes):

        # Episode "Preprocessing"
        state           = env.reset()
        constraints_old = np.zeros([constraint_dim*num_agents])

        for step in range(steps_per_episode):

            # Simulation
            action = np.random.uniform(-1, 1, num_agents*act_dim)
            action = np.split(action, num_agents)

            action_copy = copy.deepcopy(action)

            next_state, reward, done, constraints, *rest = env.step(action_copy)

            # Omit first simulation step
            if step == 0:
                constraints_old = constraints
                continue

            # Constraint diff 
            diff            = list(map(operator.sub, constraints, constraints_old))
            constraints_old = constraints

            # Store stuff to buffers for training
            idx = episode*(steps_per_episode - 1) + step - 1
            state_buf[idx,:]        = np.concatenate(state)
            actions_buf[idx,:]      = np.concatenate(action)
            constraints_diff[idx,:] = np.concatenate(diff)

            # update state
            state = next_state

            #if all(done) == True or is_collide(scenario, agents):
            #    state = env.reset()

    pd.DataFrame(state_buf).to_csv("DD_state.csv")
    pd.DataFrame(actions_buf).to_csv("DD_action.csv")
    pd.DataFrame(constraints_diff).to_csv("DD_constraints.csv")
    print("Done... Data saved")

if __name__ == "__main__":
    main()
