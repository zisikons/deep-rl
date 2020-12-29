import torch
import torch.nn as nn
import torch.functional as F
import numpy as np


class Actor(nn.Module):
    def __init__(self, state_dim=12, act_dim=2, num_agents=3, hidden_size = [128, 64]):
        super(Actor, self).__init__()
        input_size  = num_agents*state_dim
        output_size = num_agents*act_dim  

        self.linear1 = nn.Linear(input_size, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], output_size)
         
        layers = [self.linear1, self.linear2, self.linear3]
        # initialize the weights
        for layer in layers:
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = nn.ReLU()(self.linear1(state))
        x = nn.ReLU()(self.linear2(x))
        x = nn.Tanh()(self.linear3(x))
        return x
