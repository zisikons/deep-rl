#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios

from core.SafeDDPG import SafeDDPGagent
from core.Noise import OUNoise
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import copy

import matplotlib.pyplot as plt
import ipdb


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

def main():

    # Usefull Directories
    abs_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    constraint_networks_dir = abs_path + '../data/constraint_networks/'
    output_dir = abs_path + '../data/agents/SafeDDPG_soft_300/'

    # Load the simulation scenario
    scenario = scenarios.load("centralized_safe.py").Scenario()
    world    = scenario.make_world()

    # Environment Setup
    env = MultiAgentEnv(world,
                        scenario.reset_world,
                        scenario.reward,
                        scenario.observation,
                        info_callback=None,
                        constraint_callback = scenario.constraints,
                        shared_viewer = True)

    # environment properties
    state_dim, act_dim, num_agents, constraint_dim = get_env_params(env)

    # Training Parameters
    batch_size = 128
    episodes   = 1000
    steps_per_episode = 300
    soften = True

    # Define Agent
    agent = SafeDDPGagent(state_dim = state_dim,
                            act_dim = act_dim,
                     constraint_dim = constraint_dim,
                         num_agents = num_agents,
            constraint_networks_dir = constraint_networks_dir,
                             soften = soften)

    noise = OUNoise(act_dim = act_dim, num_agents = num_agents,
            act_low = -1, act_high = 1, decay_period = episodes)

    # Counters
    rewards    = []
    collisions = []
    infeasible = []
    total_collisions = 0

    # Simulation
    for episode in range(episodes):
        # Preprocessing
        state = env.reset()
        episode_reward = 0

        # Collision related
        agent.reset_metrics()
        episode_collisions = 0
        constraint = num_agents * [5*np.ones(constraint_dim)]

        for step in range(steps_per_episode):

            action = agent.get_action(np.concatenate(state), constraint)
            action = np.concatenate(action)
            action = noise.get_action(action, step, episode)
            action = np.split(action, num_agents)
            action_copy = copy.deepcopy(action) # list is mutable
            next_state, reward,done ,_ , constraint = env.step(action_copy)

            agent.memory.store(np.concatenate(state), np.concatenate(action), reward[0], np.concatenate(next_state))

            # Count collisions
            for i in range(len(env.world.agents)):
                for j in range(i + 1, len(env.world.agents), 1):
                    if scenario.is_collision(env.world.agents[i],env.world.agents[j]):
                        episode_collisions += 1

            state = next_state
            episode_reward += reward[0]
            if all(done) == True:
                print(f"Episode: {episode+1}/{episodes}, episode reward {episode_reward}, exploration {noise.sigma}")
                break
            elif step == steps_per_episode-1:
                print(f"Episode: {episode+1}/{episodes}, episode reward {episode_reward}, collisions {episode_collisions}")


            '''
            if all(done) == True:
                print(f"Episode: {episode+1}/{episodes}, episode reward {episode_reward}, exploration {noise.sigma}")
                break
            elif step == steps_per_episode-1:
                print(f"Episode: {episode+1}/{episodes}, episode reward {episode_reward}")
            '''

        if (agent.memory.ptr == agent.memory.max_size):
            print("updating agent ...")
            data = agent.get_data()
            for _ in range(200):
                agent.update(data, batch_size)

        # Save Results
        total_collisions += episode_collisions
        rewards.append(episode_reward)
        collisions.append(total_collisions)
        infeasible.append(agent.get_infeasible())

        print("Interventions =" + str(agent.get_interventions()))
        print("Problem Infeasible =" + str(agent.get_infeasible()))

    # Save Experiment results
    agent.save_params(output_dir)   # agent networks
    np.save(output_dir + 'rewards', np.array(rewards))
    np.save(output_dir + 'collisions', np.array(collisions))
    np.save(output_dir + 'infeasible', np.array(infeasible))

    # evaluating the agent's performace after training 
    rec = VideoRecorder(env, output_dir +  "policy.mp4")
    episode_length = 200
    n_eval = 10
    returns = []
    print("Evaluating agent...")

    #ipdb.set_trace()
    for i in range(n_eval):
        print(f"Testing policy: episode {i+1}/{n_eval}")
        state = env.reset()
        cumulative_return = 0
        # The environment will set terminal to True if an episode is done.
        terminal = False
        env.reset()
        for t in range(episode_length):
            if i <= 10:
                #rec.capture_frame()
                if hasattr(env.unwrapped, 'automatic_rendering_callback'):
                    env.unwrapped.automatic_rendering_callback = rec.capture_frame
                else:
                    rec.capture_frame()
            # Taking an action in the environment
            action = agent.get_action(np.concatenate(state), constraint)
            state, reward, terminal,*rest = env.step(action)
            cumulative_return += reward[0]
            if bool(np.prod(terminal)):
                break
        returns.append(cumulative_return)
        print(f"Achieved {cumulative_return:.2f} return.")
        if i == 10:
            rec.close()
            print("Saved video of 10 episodes to 'policy.mp4'.")
    env.close()
    print(f"Average return: {np.mean(returns):.2f}")
 
if __name__ == "__main__":
    main()
