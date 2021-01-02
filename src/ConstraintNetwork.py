import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optimizer

import numpy as np
import ipdb

class ConstraintNetwork(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size = 10, lr = 3e-5):
        super(ConstraintNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, act_dim)
        self.layers = [self.layer1, self.layer2]

        self.optimizer = optimizer.Adam(self.parameters(), lr)
        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
       x = nn.ReLU()(self.layer1(x))
       x = self.layer2(x)
       return x

    def train(self, state, action, constraints_diff, epochs=5, batch_size = 256, split_ratio = 0.10):

       shuffle_idx = np.arange(state.shape[0])
       np.random.shuffle(shuffle_idx)
       split_idx =  int(state.shape[0]*split_ratio)
       train_idx = shuffle_idx[split_idx:]
       val_idx   = shuffle_idx[0:split_idx:]


       # Train data
       train_state       = state[train_idx,:]
       train_action      = action[train_idx,:]
       train_constraints_diff = constraints_diff[train_idx]

       # Validation data
       val_state            = state[val_idx,:]
       val_action           = action[val_idx,:]
       val_constraints_diff = constraints_diff[val_idx]

       for epoch in range(epochs):
           for batch in range(train_state.shape[0]//batch_size):
                batch_idx         = np.random.choice(np.arange(train_state.shape[0]), size = batch_size)

                state_batch            = torch.Tensor(train_state[batch_idx,:])
                action_batch           = torch.Tensor(train_action[batch_idx,:])
                constraints_diff_batch = torch.Tensor(train_constraints_diff[batch_idx])

                out  = self.forward(state_batch)
                assert out.requires_grad == True
                assert out.shape[0] == batch_size

                dot_prod = torch.sum(torch.mul(out, action_batch), axis = 1)

                loss = nn.MSELoss()(constraints_diff_batch, dot_prod)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

           val_state            = torch.Tensor(val_state)
           val_action           = torch.Tensor(val_action)
           val_constraints_diff = torch.Tensor(val_constraints_diff)

           with torch.no_grad():
                out  = self.forward(val_state)
                dot_prod = torch.sum(torch.mul(out, val_action), axis = 1)
                loss = nn.MSELoss()(val_constraints_diff,dot_prod)
           print(f"Epoch: {epoch+1}/{epochs}, val_loss {loss}")

def main():

    import pandas as pd
    import ipdb

    # Import Dataset
    state            = pd.read_csv("DD_state.csv").to_numpy()
    action           = pd.read_csv("DD_action.csv").to_numpy()
    constraints_diff = pd.read_csv("DD_constraints.csv").to_numpy()

    # Remove Indices
    state = state[:,1:]
    action = action[:,1:]
    constraints_diff = constraints_diff[:,1:]

    # Number of networks 
    N = constraints_diff.shape[1]

    for i in range(N):

        # Define Network for constraint i
        net = ConstraintNetwork(state_dim = state.shape[1], act_dim = action.shape[1])
        net.train(state, action, constraints_diff[:,i])

        # Store
        torch.save(net.state_dict(),"constraint_net" + str(i+1) + ".pkl")



if __name__ == "__main__":
    main()
























