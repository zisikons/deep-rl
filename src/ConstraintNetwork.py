import torch 
import torch.nn as nn
import torch.functional as F
import torch.optim as optimizer

import numpy as np
import ipdb
class ConstraintNetwork(nn.Module):
    def __init__(self, state_dim=14, act_dim=2, hidden_size = 10, lr = 3e-5):
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
   
    def train(self, state, action, next_state, epochs=25, batch_size = 256, split_ratio = 0.25):
       
       shuffle_idx = np.arange(state.shape[0]) 
       np.random.shuffle(shuffle_idx)
       split_idx =  int(state.shape[0]*split_ratio)
       train_idx = shuffle_idx[split_idx:]
       val_idx   = shuffle_idx[0:split_idx:]
       
       # Train data
       train_state      = state[train_idx,:]
       train_action     = action[train_idx,:]
       train_next_state = next_state[train_idx,:]
       
       # Validation data
       val_state      = state[val_idx,:]
       val_action     = action[val_idx,:]
       val_next_state = next_state[val_idx,:] 
        
       for epoch in range(epochs):
           for batch in range(train_state.shape[0]//batch_size):
                batch_idx        = np.random.choice(np.arange(train_state.shape[0]), size = batch_size)
                state_batch      = torch.Tensor(train_state[batch_idx,:])
                action_batch     = torch.Tensor(train_action[batch_idx,:])
                next_state_batch = torch.Tensor(train_next_state[batch_idx,:])

                out  = self.forward(state_batch)
                assert out.requires_grad == True
                assert out.shape[0] == batch_size
               
                dot_prod = torch.sum(torch.mul(out, action_batch), axis = 1)
                loss = nn.MSELoss()(self.constraint(next_state_batch), self.constraint(state_batch) + dot_prod)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
           val_state      = torch.Tensor(val_state)
           val_action     = torch.Tensor(val_action)
           val_next_state = torch.Tensor(val_next_state)
            
           with torch.no_grad():
                out  = self.forward(val_state) 
                dot_prod = torch.sum(torch.mul(out, val_action), axis = 1)
                loss = nn.MSELoss()(self.constraint(val_next_state), self.constraint(val_state) + dot_prod)
           print(f"Epoch: {epoch+1}/{epochs}, val_loss {loss}")

 
                
    def constraint(self,state, col_margin = 0.1):
        constraint_val1 = -(torch.sum(torch.mul(state[:, -2:], state[:, -2:]), axis = 1) - col_margin**2)
        constraint_val2 = -(torch.sum(torch.mul(state[:, -4:-2], state[:, -4:-2]), axis = 1) - col_margin**2)
        return constraint_val1
        

def main():

    import pandas as pd
    import ipdb
    

    state = pd.read_csv("D_state.csv").to_numpy()
    action = pd.read_csv("D_action.csv").to_numpy()
    next_state = pd.read_csv("D_next_state.csv").to_numpy()
    
    state = state[:,1:]
    action = action[:,1:]
    next_state = next_state[:,1:]
    
    # split the dataset into agents
    states = np.split(state, 3 , axis = 1)
    actions = np.split(action, 3 , axis = 1)
    next_states = np.split(next_state, 3 , axis = 1)
    net = ConstraintNetwork(state_dim = 14, act_dim = 2)
    
    '''
    1)1-2, 2)2-3, 3)3-1

    1) network1(state1) --> 1-2
    2) network2(stat2) --> 2-3
    3) network1(state3) -->3-1
    
    #state_agent1 = [my_pos, my_vel, other_pos1-my_pos, other_pos2-my_pos]
    constraint1 = norm(other_pos1-my_pos)**2 -margin**2
    constaint2  = norm(other_pos2-my_pos)**2 -margin**2
    state_agent2 = [my_pos, my_vel, other_pos1-my_pos, other_pos2-my_pos]
    '''

    for i in range(3):
        print("passing agent " + str(i+1) + "...")
        net.train(states[i], actions[i], next_states[i])
    torch.save(net.state_dict(),"constraint_net_2.pkl")


    ipdb.set_trace()


if __name__ == "__main__":
    main()
























