#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np
from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios

from core.DDPG import DDPGagent
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

    num_agents = len(state_space)

    assert num_agents == len(action_space)
    return state_dim, action_dim, num_agents

def main():

    # Usefull Directories
    abs_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    constraint_networks_dir = abs_path + '../data/constraint_networks/'
    output_dir = abs_path + '../data/agents/DDPG/'

    # Load the simulation scenario
    scenario = scenarios.load("centralized_safe.py").Scenario()
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

    # environment properties
    state_dim, act_dim, num_agents = get_env_params(env)

    # Training Parameters
    batch_size = 128
    episodes = 2000
    steps_per_episode = 200


    agent = DDPGagent(state_dim = state_dim, act_dim = act_dim, num_agents = num_agents)
    noise = OUNoise(act_dim = act_dim, num_agents = num_agents, act_low = -1, act_high = 1, decay_period = episodes)

    rewards = []
    collisions = []
    total_collisions = 0

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_collisions = 0

        for step in range(steps_per_episode):

            action = agent.get_action(np.concatenate(state))
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
                print(f"Episode: {episode+1}/{episodes}, episode reward {episode_reward}")
                break
            elif step == steps_per_episode-1:
                print(f"Episode: {episode+1}/{episodes}, episode reward {episode_reward}")

        if (episode != 0):
            if(episode%100 == 0):
                print("updating agent ...")
                data = agent.get_data()
                for _ in range(200):
                    agent.update(data, batch_size)

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
    episode_length = 200
    n_eval = 10
    returns = []
    print("Evaluating agent...")

    for i in range(n_eval):
        print(f"Testing policy: episode {i+1}/{n_eval}")
        state = env.reset()
        cumulative_return = 0
        # The environment will set terminal to True if an episode is done.
        env.reset()
        for t in range(episode_length):
            if i <= 10:
                if hasattr(env.unwrapped, 'automatic_rendering_callback'):
                    env.unwrapped.automatic_rendering_callback = rec.capture_frame
                else:
                    rec.capture_frame()
            # Taking an action in the environment
            action = agent.get_action(np.concatenate(state))
            state, reward, done,*rest = env.step(action)
            cumulative_return += reward[0]
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
