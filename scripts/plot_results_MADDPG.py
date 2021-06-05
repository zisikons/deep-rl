#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Usefull Directories
    abs_path   = os.path.dirname(os.path.abspath(__file__)) + '/'
    target_dir = abs_path + '../data/plots/'

    # Export Results for training
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    true_run = True
    N = 1000

    safe_maddpg_soft = abs_path + '../data/agents/SafeMADDPG_soft/'
    safe_maddpg_hard = abs_path + '../data/agents/SafeMADDPG_hard/'
    maddpg_vanilla   = abs_path + '../data/agents/MADDPG/'


    # Import Results
    reward_soft = np.load(safe_maddpg_soft + 'rewards.npy')[0:N:10]
    reward_hard = np.load(safe_maddpg_hard + 'rewards.npy')[0:N:10]
    reward_maddpg = np.load(maddpg_vanilla + 'rewards.npy')[0:N:10]

    collisions_soft = np.load(safe_maddpg_soft + 'collisions.npy')[0:N]
    collisions_hard = np.load(safe_maddpg_hard + 'collisions.npy')[0:N]
    collisions_maddpg = np.load(maddpg_vanilla + 'collisions.npy')[0:N]

    infeasibilities_soft = np.load(safe_maddpg_soft + 'infeasible.npy')[0:N]
    infeasibilities_hard = np.load(safe_maddpg_hard + 'infeasible.npy')[0:N]

    # Plot Type 1: Rewards
    reward_fig = plt.figure()
    plt.plot(reward_soft)
    plt.plot(reward_hard)
    plt.plot(reward_maddpg)
    reward_fig.suptitle('Rewards')
    plt.xlabel('Episodes')
    plt.legend(['Soft MADDPG', 'Hard MADDPG', 'MADDPG'])
    plt.show()
    reward_fig.savefig(target_dir + 'rewards.pdf', bbox_inches='tight')

    # Plot Type 2: Collision
    collision_fig = plt.figure()
    plt.plot(collisions_soft)
    plt.plot(collisions_hard)
    plt.plot(collisions_maddpg)
    collision_fig.suptitle('Collisions')
    plt.xlabel('Episodes')
    plt.legend(['Soft MADDPG', 'Hard MADDPG', 'MADDPG'])
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
