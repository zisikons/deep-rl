import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from core.constraint_network import ConstraintNetwork
import numpy as np
import torch 

torch.manual_seed(2021) 
np.random.seed(2021)

def main():

    # Settings
    abs_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    datasets_dir = abs_path + '../data/'
    output_dir   = abs_path + '../data/constraint_networks_MADDPG/'

    # Training Settings
    EPOCHS = 100
    BATCH_SIZE = 256
    VAL_RATIO = 0.1

    # Check if output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Import Datasets 
    state = np.genfromtxt(datasets_dir + "D_state_decentralized.csv", delimiter = ',')
    action = np.genfromtxt(datasets_dir + "D_action_decentralized.csv", delimiter = ',')
    constraint_diff = np.genfromtxt(datasets_dir + "D_constraint_decentralized.csv", delimiter = ',')
    # Remove Indices
    state  = state[:, 1:]
    action = action[:, 1:]
    constraint_diff = constraint_diff[:, 1:]

    # Number of networks 
    N = constraint_diff.shape[1]

    # Train one network for each constraint
    for i in range(N):

        # Define Network for constraint i
        net = ConstraintNetwork(state_dim = state.shape[1], act_dim = action.shape[1])
        net.train(state, action, constraint_diff[:, i], EPOCHS, BATCH_SIZE, VAL_RATIO)

        # Store
        torch.save(net.state_dict(), output_dir + "constraint_net_" + str(i) + ".pkl")

if __name__ == "__main__":
    main()
