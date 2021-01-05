#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import ipdb

def main():
    # Usefull Directories
    abs_path   = os.path.dirname(os.path.abspath(__file__)) + '/'
    target_dir = abs_path + '../data/plots/'

    # Export Results for training
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    true_run = True
    N = 8000

    # Experiment Directories
    if true_run:
        safe_ddpg_soft = abs_path + '../data/agents/SafeDDPG_soft/'
        safe_ddpg_hard = abs_path + '../data/agents/SafeDDPG_hard/'
        ddpg_vanilla   = abs_path + '../data/agents/DDPG/'

        # Import Results
        reward_soft = np.load(safe_ddpg_soft + 'rewards.npy')[0:N]
        reward_hard = np.load(safe_ddpg_hard + 'rewards.npy')[0:N]
        reward_ddpg = np.load(ddpg_vanilla + 'rewards.npy')[0:N]

        collisions_soft = np.load(safe_ddpg_soft + 'collisions.npy')[0:N]
        collisions_hard = np.load(safe_ddpg_hard + 'collisions.npy')[0:N]
        collisions_ddpg = np.load(ddpg_vanilla + 'collisions.npy')[0:N]

        infeasibilities_soft = np.load(safe_ddpg_soft + 'infeasible.npy')[0:N]
        infeasibilities_hard = np.load(safe_ddpg_hard + 'infeasible.npy')[0:N]
    else:
        safe_ddpg_soft = '../data/test/'
        safe_ddpg_hard = '../data/test/'
        ddpg_vanilla   = '../data/test/'

        # Import Results
        reward_soft = np.load(safe_ddpg_soft + 'rewards_soft.npy')[0:N]
        reward_hard = np.load(safe_ddpg_hard + 'rewards_hard.npy')[0:N]
        reward_ddpg = np.load(ddpg_vanilla + 'rewards_ddpg.npy')[0:N]

        collisions_soft = np.load(safe_ddpg_soft + 'collisions_soft.npy')[0:N]
        collisions_hard = np.load(safe_ddpg_hard + 'collisions_hard.npy')[0:N]
        collisions_ddpg = np.load(ddpg_vanilla + 'collisions_ddpg.npy')[0:N]

        infeasibilities_soft = np.load(safe_ddpg_soft + 'infeasible_soft.npy')[0:N]
        infeasibilities_hard = np.load(safe_ddpg_hard + 'infeasible_hard.npy')[0:N]

    # Plot Type 1: Rewards
    reward_fig = plt.figure()
    plt.plot(reward_soft)
    plt.plot(reward_hard)
    plt.plot(reward_ddpg)
    reward_fig.suptitle('Rewards')
    plt.xlabel('Episodes')
    plt.legend(['Soft DDPG', 'Hard DDPG', 'DDPG'])
    plt.show()
    reward_fig.savefig(target_dir + 'rewards.pdf', bbox_inches='tight')

    # Plot Type 2: Collision
    collision_fig = plt.figure()
    plt.plot(collisions_soft)
    plt.plot(collisions_hard)
    plt.plot(collisions_ddpg)
    collision_fig.suptitle('Collisions')
    plt.xlabel('Episodes')
    plt.legend(['Soft DDPG', 'Hard DDPG', 'DDPG'])
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
