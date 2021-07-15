#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def main():
    # Usefull Directories
    abs_path   = os.path.dirname(os.path.abspath(__file__)) + '/'
    target_dir = abs_path + '../data/plots/'

    # Export Results for training
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    true_run = True
    N = 100

    seed_range = np.arange(10,11)

    safe_maddpg_soft = abs_path + '../data/agents/SafeMADDPG_soft_/'
    safe_maddpg_hard = abs_path + '../data/agents/SafeMADDPG_hard/'
    maddpg_vanilla   = abs_path + '../data/agents/MADDPG/'
     
    rewards_soft    = []
    rewards_hard    = []
    rewards_vanilla = []
    
    infeasibilities_soft = []
    infeasibilities_hard = []
    for seed in seed_range:

        # Import Results
        #rewards_soft.append(np.load(safe_maddpg_soft +"seed" + str(seed) + "/" + 'rewards.npy')[0:N:10])
        rewards_hard.append(np.load(safe_maddpg_hard + "seed" + str(seed) + "_new/"+'rewards.npy')[0:N])
        #rewards_maddpg.append(np.load(maddpg_vanilla + "seed" + str(seed) + "/" + 'rewards.npy')[0:N:10])

        collisions_soft = np.load(safe_maddpg_soft + "seed" + str(seed) + "_new/" + 'collisions.npy')[0:N]
        collisions_hard = np.load(safe_maddpg_hard + "seed" + str(seed) + "_new/" + 'collisions.npy')[0:N]
        #collisions_maddpg = np.load(maddpg_vanilla + "seed" + str(seed) + "/" +'collisions.npy')[0:N]

        #infeasibilities_soft.append(np.load(safe_maddpg_soft + "seed" + str(seed) + "/" +'infeasible.npy')[0:N])
        infeasibilities_hard.append(np.load(safe_maddpg_hard + "seed" + str(seed) + "_new/" +'infeasible.npy')[0:N])
    
    columns = ["episodes"] + ["seed" + str(i) for i in seed_range]
    episode_index = np.array([int(i+1) for i in np.arange(0, N)]).reshape(N,1)
    rewards_hard = np.array(rewards_hard).T
    rewards_hard = np.concatenate([episode_index, rewards_hard], axis = 1)
    rewards_hard = pd.DataFrame(rewards_hard, columns = columns)  
    # Plot Type 1: Rewards
    reward_fig = plt.figure()
    #plt.plot(reward_soft)
    plt.plot(rewards_hard)
    #plt.plot(reward_maddpg)
    reward_fig.suptitle('Rewards')
    plt.xlabel('Episodes')
    plt.legend(['Soft MADDPG', 'Hard MADDPG', 'MADDPG'])
    plt.show()
    reward_fig.savefig(target_dir + 'rewards.pdf', bbox_inches='tight')

    # Plot Type 2: Collision
    collision_fig = plt.figure()
    plt.plot(collisions_soft)
    plt.plot(collisions_hard)
    #plt.plot(collisions_maddpg)
    collision_fig.suptitle('Collisions')
    plt.xlabel('Episodes')
    #plt.legend(['Soft MADDPG', 'Hard MADDPG', 'MADDPG'])
    plt.legend(['Soft MADDPG', 'Hard MADDPG'])
    plt.show()
    collision_fig.savefig(target_dir +'collisions.pdf', bbox_inches='tight')

    # Plot Type 3: Infeasible
    infeasibility_fig = plt.figure()
    plt.scatter(np.arange(N), infeasibilities_soft)
    plt.scatter(np.arange(N), infeasibilities_hard)
    infeasibility_fig.suptitle('Infeasibility Occurances')
    plt.xlabel('Episodes')
    plt.legend(['Soft DDPG', 'Hard DDPG'])
    plt.show()
    infeasibility_fig.savefig(target_dir +'infeasibility.pdf', bbox_inches='tight')


if __name__ == "__main__":
    main()
