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
import ipdb

def main():
    try:
        seed = int(sys.argv[1])
        torch.manual_seed(seed)
        np.random.seed(seed)

    except: 
        seed = '' 

    # Usefull Directories
    abs_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    constraint_networks_dir = abs_path + '../data/constraint_networks_MADDPG/'
    output_dir = abs_path + '../data/agents/SafeMADDPG_soft/'+ "seed" + str(seed) + '_dist/'

    # Load the simulation scenario
    safe_initialization = False
    apply_disturbance   = True
    
    scenario = scenarios.load("decentralized_safe.py").Scenario(safe_initialization)
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
    disturbance_range = {'lower':-1, 'upper':1}
    print(env_params) 

    # Training Parameters
    batch_size = 128
    episodes = 8000
    steps_per_episode = 300
    agent_update_rate = 100  # update agent every
    agent_updates     = 50   # number of sampled batches
    
    # MADDPG Agent
    agent = SafeMADDPGagent(state_dim = state_dim,
                            act_dim = act_dim,
                            N_agents = N_agents,
                            batch_size = batch_size,
                            constraint_dim = constraint_dim,
                            constraint_networks_dir=constraint_networks_dir)

    # Exploratory noise
    noise = OUNoise(act_dim = act_dim,num_agents=N_agents, act_low = -1, act_high = 1, decay_period = episodes)

    rewards = []
    collisions = []
    infeasible = []
    total_collisions = 0

    for episode in range(episodes):
        # Preprocessing
        state = env.reset()
        episode_reward = 0

        # Collision related
        agent.reset_metrics()
        episode_collisions = 0
        constraint = N_agents * [5*np.ones(constraint_dim)]

        for step in range(steps_per_episode):

            # Compute safe action
            action = agent.get_action(state,constraint)

            # Add exploration noise
            action = np.concatenate(action)
            action = noise.get_action(action, step, episode)
            action = np.split(action, N_agents)
            
            # correct the action here might be better?
            action,_ = agent.correct_actions(state, action, constraint)
            if apply_disturbance:
                # apply disturbance
                disturbance = (disturbance_range['upper'] - disturbance_range['lower'])*np.random.rand(N_agents*act_dim) + disturbance_range['lower']
                action = action + disturbance
            action = np.split(action, N_agents)
            
            # Feed the action to the environment
            action_copy = copy.deepcopy(action) # list is mutable
            next_state, reward, done ,_ , constraint = env.step(action_copy)
            
            agent.memory.store(state, action, reward, next_state)
            # Count collisions
            for i in range(len(env.world.agents)):
                for j in range(i + 1, len(env.world.agents), 1):
                    if scenario.is_collision(env.world.agents[i],env.world.agents[j]):
                        episode_collisions += 1


            # Check if episode terminates
            if all(done) == True or step == steps_per_episode-1:
                print(f"Episode: {episode+1}/{episodes}, \
                        episode reward {episode_reward}, \
                        collisions {episode_collisions}")
                break
            # Prepare Next iteration
            state = next_state
            episode_reward += (sum(reward)/N_agents) # average reward over all agents
        # Update Agents every # episodes
        if(episode % agent_update_rate == 0 and episode > 0):
            # Perform 200 updates (for the time fixed)
            print("updating agent ...")
            for _ in range(agent_updates):
                agent.update()
            print("done")

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


if __name__ == "__main__":
    main()
