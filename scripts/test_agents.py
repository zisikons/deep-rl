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
from core.MADDPG import MADDPGagent
from core.Noise import OUNoise

from gym.wrappers.monitoring.video_recorder import VideoRecorder

from inspect import signature
import ipdb

def main():
    try:
        seed = int(sys.argv[1])
    except:
        print("add a seed argument when running the file. Must be a positive integer.")
        return
    torch.manual_seed(seed)
    np.random.seed(seed)


    agents_names = ["SafeMADDPG_soft", "SafeMADDPG_soft_reward", "SafeMADDPG_hard", "MADDPG"]  
    agent_paths  = []  
    # Usefull Directories
    abs_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    constraint_networks_dir           = abs_path + '../data/constraint_networks_MADDPG/'
    agent_names = ["SafeMADDPG_soft", "SafeMADDPG_soft_reward", "SafeMADDPG_hard", "MADDPG"]  
    agent_paths  = [abs_path + '../data/agents/SafeMADDPG_soft/'+ "seed" + str(seed) + '/', 
                    abs_path + '../data/agents/SafeMADDPG_soft_reward/'+ "seed" + str(seed) + '/',
                    abs_path + '../data/agents/SafeMADDPG_hard/'+ "seed" + str(seed) + '/',
                    abs_path + '../data/agents/MADDPG/'+ "seed" + str(seed) + '/']

    output_dirs = dict(zip(agent_names, agent_paths))
    
    # set up the environment
    scenario = scenarios.load("decentralized_safe.py").Scenario()
    world    = scenario.make_world()
    
    envs = [MultiAgentEnv(world,
                        scenario.reset_world,
                        scenario.reward,
                        scenario.observation,
                        info_callback=None,
                        done_callback = scenario.done,
                        constraint_callback = scenario.constraints,
                        shared_viewer = True) for _ in range(len(agent_names))]
    envs = dict(zip(agent_names,envs))

    # get the scenario parameters
    env_params = envs[agent_names[0]].get_env_parameters()
    state_dim = env_params["state_dim"]
    act_dim   = env_params["act_dim"]
    constraint_dim = env_params["constraint_dim"]
    N_agents = env_params["num_agents"]
    print(env_params)

    # Training Parameters
    batch_size = 128
    episodes = 8000
    steps_per_episode = 300
    agent_update_rate = 100 # update agent every # episodes old:100
    
    # Load agents
    agents = []
    
    # soft agent
    agent_soft = SafeMADDPGagent(state_dim = state_dim,
                                      act_dim = act_dim,
                                      N_agents = N_agents,
                                      batch_size = batch_size,
                                      constraint_dim = constraint_dim,
                                      constraint_networks_dir=constraint_networks_dir)
    agent_soft.load_params(output_dirs['SafeMADDPG_soft'])
    agents.append(agent_soft) 
    
    #soft reward agent
    agent_soft_reward  = SafeMADDPGagent(state_dim = state_dim,
                                    act_dim = act_dim,
                                    N_agents = N_agents,
                            batch_size = batch_size,
                            constraint_dim = constraint_dim,
                            constraint_networks_dir=constraint_networks_dir)

    agent_soft_reward.load_params(output_dirs['SafeMADDPG_soft_reward'])
    agents.append(agent_soft_reward)
    # hard agent
    agent_hard = SafeMADDPGagent(state_dim = state_dim,
                           act_dim = act_dim,
                           N_agents = N_agents,
                           batch_size = batch_size,
                           constraint_dim = constraint_dim,
                           constraint_networks_dir=constraint_networks_dir,
                           soften = False)
    agent_hard.load_params(output_dirs['SafeMADDPG_hard'])
    agents.append(agent_hard) 
    # vanilla agent
    agent_vanilla = MADDPGagent(state_dim = state_dim,
                        act_dim = act_dim,
                        N_agents = N_agents,
                        critic_state_mask = np.arange(state_dim).tolist(),
                        batch_size = batch_size) 
    agent_vanilla.load_params(output_dirs['MADDPG'])
    agents.append(agent_vanilla)
    agents = dict(zip(agent_names, agents))
    for agent_name, agent in agents.items():
        
        env = envs[agent_name]
        # evaluating the agent's performace after training 
        #rec = VideoRecorder(env, output_dirs[agent_name] +  "test_policy.mp4")
        n_eval = 100
        returns = []
        episodes_collisions = []
        print("Evaluating agent...")
        constraint = N_agents * [10*np.ones(constraint_dim)]
        for i in range(n_eval):
            print(f"Testing policy for {agent_name} : episode {i+1}/{n_eval}")
            state = env.reset()
            cumulative_return = 0
            episode_collisions = 0
            env.reset()
            for t in range(steps_per_episode):
                # Taking an action in the environment
                result = agent.get_action(state, constraint)
                if type(result) == tuple:
                    action = result[0]
                else:
                    action = result
                action_copy = copy.deepcopy(action)
                next_state, reward,done ,_ , constraint = env.step(action_copy)
                cumulative_return += sum(reward)/N_agents

                # update state 
                state = next_state
                
                # Count collisions
                for k in range(len(env.world.agents)):
                    for j in range(k + 1, len(env.world.agents), 1):
                        if scenario.is_collision(env.world.agents[k],env.world.agents[j]):
                            episode_collisions += 1

                
                if all(done) == True:
                    break
            returns.append(cumulative_return)
            episodes_collisions.append(episode_collisions)
            print(f"Achieved {cumulative_return:.2f} return.")
            if i == n_eval-1:
                env.close()
                print(f"Average return: {np.mean(returns):.2f}")
                episodes_collisions = np.array(episodes_collisions)
                np.save(output_dirs[agent_name]+"test_collisions.npy", episodes_collisions)

if __name__ == "__main__":
    main()
