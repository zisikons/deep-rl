import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

class Critic(nn.Module):
    def __init__(self, state_dim = 12, act_dim = 2 ,num_agents = 3,hidden_size = [500, 500]):
        super(Critic, self).__init__()

        input_size  = state_dim*num_agents + act_dim*num_agents
        output_size = 1

        self.linear1 = nn.Linear(input_size, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], output_size)
        layers = [self.linear1, self.linear2, self.linear3]
        
        # initialize the weights
        for layer in layers:
            torch.nn.init.xavier_uniform_(layer.weight)
 
    
    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = nn.ReLU()(self.linear1(x))
        x = nn.ReLU()(self.linear2(x))
        x = self.linear3(x)
        return x

