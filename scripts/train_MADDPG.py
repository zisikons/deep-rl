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

from core.MADDPG import MADDPGagent
from core.Noise import OUNoise

from gym.wrappers.monitoring.video_recorder import VideoRecorder
import ipdb

def main():

    try:
        seed = int(sys.argv[1])
    except:
        print("add a seed argument when running the file. Must be a positive integer.")
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
 
    # Usefull Directories
    abs_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    constraint_networks_dir = abs_path + '../data/constraint_networks_MADDPG/'
    output_dir = abs_path + '../data/agents/MADDPG/' + "seed" + str(seed) + '_dist/'
 
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
    num_agents = env_params["num_agents"]
    print(env_params) 
    # Training Parameters
    batch_size = 128
    episodes = 8000
    steps_per_episode = 300
    agent_update_rate = 100 


    # MADDPG Agent
    agent = MADDPGagent(state_dim = state_dim,
                        act_dim = act_dim,
                        N_agents = num_agents,
                        critic_state_mask = np.arange(state_dim).tolist(),
                        batch_size = batch_size)

    # Will stay as is or?
    noise = OUNoise(act_dim = act_dim, num_agents = num_agents, act_low = -1, act_high = 1, decay_period = episodes)

    rewards = []
    collisions = []
    total_collisions = 0

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_collisions = 0
        for step in range(steps_per_episode):

            # Compute action
            action = agent.get_action(state)

            # Add exploration noise
            action = np.concatenate(action)
            action = noise.get_action(action, step, episode)
            
            # apply disturbance
            disturbance = 2*np.random.rand(num_agents*act_dim)-1
            action = action + disturbance

            action = np.split(action, num_agents)

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
            episode_reward += reward[0] # will all agents have the same reward?

        # Update Agents every # episodes
        if(episode % agent_update_rate == 0 and episode > 0):
            # Perform 200 updates (for the time fixed)
            print("updating agent ...")
            for _ in range(50):
                agent.update()
            print("done")

        # Save Results
        total_collisions += episode_collisions
        rewards.append(episode_reward)
        collisions.append(total_collisions)

    # Save Experiment results
    agent.save_params(output_dir)   # agent networks
    np.save(output_dir + 'rewards', np.array(rewards))
    np.save(output_dir + 'collisions', np.array(collisions))

    # evaluating the agent's performace after training 
    rec = VideoRecorder(env, output_dir +  "policy.mp4")
    episode_length = steps_per_episode
    n_eval = 10
    returns = []
    print("Evaluating agent...")

    for i in range(n_eval):
        print(f"Testing policy: episode {i+1}/{n_eval}")
        state = env.reset()
        cumulative_return = 0
        env.reset()
        for t in range(episode_length):
            if i <= 10:
                if hasattr(env.unwrapped, 'automatic_rendering_callback'):
                    env.unwrapped.automatic_rendering_callback = rec.capture_frame
                else:
                    rec.capture_frame()
            # Taking an action in the environment
            action = agent.get_action(state)
            action_copy = copy.deepcopy(action)
            next_state, reward,done ,_ , constraint = env.step(action_copy)
            cumulative_return += reward[0]
            
            # update state 
            state = next_state

            if all(done) == True:
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
