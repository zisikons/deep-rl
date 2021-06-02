import os
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optimizer

import numpy as np
import pandas as pd

class ConstraintNetwork(nn.Module):
    '''
    Neural Network used to learn the constraint sensitivity.
    (Tailored to this specific set of examples)
    '''

    def __init__(self, state_dim, act_dim, hidden_size = 10, lr = 5e-4):
        '''
        Constructor
        Arguments:
            - state_dim   : the state dimension of the RL agent
            - action_dim  : the action dimension of the RL agent
            - hidden_size : hidden layer size
        '''
        super(ConstraintNetwork, self).__init__()

        # Network Architecture
        self.layer_1 = nn.Linear(state_dim, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, act_dim)
        self.layers  = [self.layer_1, self.layer_2]

        # Optimizer
        self.optimizer = optimizer.Adam(self.parameters(), lr)

        # Initialization
        self.init_weights()

    def init_weights(self):
        '''
        Weights initialization method
        '''
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
       '''
       Forward propagation method
       '''
       x = nn.ReLU()(self.layer_1(x))
       x = self.layer_2(x)
       return x

    def train(self, state, action, constraints_diff, epochs=100, batch_size = 256, split_ratio = 0.10):

        # Training - Validation split 
        shuffle_idx = np.arange(state.shape[0])
        np.random.shuffle(shuffle_idx)

        split_idx =  int(state.shape[0]*split_ratio)
        train_idx = shuffle_idx[split_idx:]
        val_idx   = shuffle_idx[0:split_idx:]

        # Training data
        train_state       = state[train_idx,:]
        train_action      = action[train_idx,:]
        train_constraints_diff = constraints_diff[train_idx]

        # Validation data
        val_state            = state[val_idx,:]
        val_action           = action[val_idx,:]
        val_constraints_diff = constraints_diff[val_idx]

        # Training Loop
        for epoch in range(epochs):
            for batch in range(train_state.shape[0]//batch_size):
                # Randomly select a batch
                batch_idx = np.random.choice(np.arange(train_state.shape[0]), size = batch_size)

                state_batch            = torch.Tensor(train_state[batch_idx,:])
                action_batch           = torch.Tensor(train_action[batch_idx,:])
                constraints_diff_batch = torch.Tensor(train_constraints_diff[batch_idx])

                out  = self.forward(state_batch)
                assert out.requires_grad == True
                assert out.shape[0] == batch_size

                # Loss Function
                dot_prod = torch.sum(torch.mul(out, action_batch), axis = 1)

                loss = nn.MSELoss()(constraints_diff_batch, dot_prod)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Evaluate NNs performance on the validation set
            val_state            = torch.Tensor(val_state)
            val_action           = torch.Tensor(val_action)
            val_constraints_diff = torch.Tensor(val_constraints_diff)

            with torch.no_grad():
                out  = self.forward(val_state)
                dot_prod = torch.sum(torch.mul(out, val_action), axis = 1)
                loss = nn.MSELoss()(val_constraints_diff,dot_prod)
            print(f"Epoch: {epoch+1}/{epochs}, val_loss {loss}")

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
    state           = pd.read_csv(datasets_dir + "D_state_decentralized.csv").to_numpy()
    action          = pd.read_csv(datasets_dir + "D_action_decentralized.csv").to_numpy()
    constraint_diff = pd.read_csv(datasets_dir + "D_constraint_decentralized.csv").to_numpy()

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

