import os
import numpy as np
import ipdb

import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn

from torch.autograd import Variable

class ReplayBuffer:
    """
    Buffer to store trajectories.
    """
    def __init__(self, state_dim, act_dim, num_agents, size):

        self.state_buf      = list()
        self.act_buf        = list()
        self.rew_buf        = list()
        self.next_state_buf = list()
        self.ptr, self.max_size = 0, size

    def store(self, state, act, rew, next_state):
        """
        Append a single timestep to the buffer. This is called at each environment
        update to store the observed outcome.
        """
        # buffer has to have room so you can store
        if self.ptr == self.max_size:
            self.state_buf.pop(0)
            self.act_buf.pop(0)
            self.rew_buf.pop(0)
            self.next_state_buf.pop(0)
            self.ptr -= 1

        # Environment related, subject to change
        self.state_buf.append(np.expand_dims(state, axis = 0))
        self.act_buf.append(np.expand_dims(act, axis = 0))
        self.rew_buf.append(np.array(rew, ndmin = 1))
        self.next_state_buf.append(np.expand_dims(next_state, axis = 0))
        self.ptr += 1

    def get(self):
        """
        Call when updating the agent networks
        """
        data = dict(state= np.concatenate(self.state_buf), act=np.concatenate(self.act_buf),
                    rew=np.concatenate(self.rew_buf), next_state = np.concatenate(self.next_state_buf))

        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

    def parse_observation(self,):
        pass


class Critic(nn.Module):
    """
    Critic neural network (centralized)
    """
    def __init__(self, N_agents = 3, state_dim = 4, act_dim = 2, hidden_size = [500, 500]):
        """
        Constructor
        Arguments:
            - state_dim   : the state dimension of the RL agent
            - action_dim  : the action dimension of the RL agent
            - hidden_size : hidden layer size
        """
        super(Critic, self).__init__()

        # Dimensions
        self.input_size  = N_agents * (state_dim + act_dim)
        self.hidden_size = hidden_size
        self.output_size = 1

        # Network Architecture
        self.layers = []

        # Step 1: Input Layer
        self.layers.append(nn.Linear(self.input_size, self.hidden_size[0]).double())

        # Step 2: Intermediate Layers
        for i in range(len(hidden_size) - 1):
            self.layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]).double())

        # Step 3: Add last layer
        self.layers.append(nn.Linear(hidden_size[-1], self.output_size).double())

        # initialize the weights
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(layer.weight)


    def forward(self, state, action):
        """
        Forward propagation method
        """
        # Params state and actions are torch tensors
        x = torch.cat([state, action], 0) #2nd index = dim of concat

        # Intermediate Layers
        for layer in self.layers[:-1]:
            x = nn.ReLU()(layer(x))

        # Last layer (linear)
        x = self.layers[-1](x)

        return x

class Actor(nn.Module):
    def __init__(self, state_dim=4, act_dim=2,  hidden_size = [100, 100]):
        super(Actor, self).__init__()

        self.input_size  = state_dim
        self.output_size = act_dim
        self.hidden_size = hidden_size

        # Network Architecture
        self.layers = []

        # Input Layer
        self.layers.append(nn.Linear(self.input_size, self.hidden_size[0]).double())

        # Intermediate Layers
        for i in range(len(hidden_size) - 1):
            self.layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]).double())

        # Last layer
        self.layers.append(nn.Linear(hidden_size[-1], self.output_size).double())

        # initialize the weights
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, state):
        """
        Param state is a torch tensor
        """

        x = state # Careful: deepcopy bug?
        # Intermediate Layers
        for layer in self.layers[:-1]:

            x = nn.ReLU()(layer(x))

        x = nn.Tanh()(self.layers[-1](x))
        return x

class MADDPG:

    def __init__(self, N_agents, state_dim, act_dim, actor_learning_rate=1e-4,
                 critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=16000,
                 hidden_size_critic = [500, 500], hidden_size_actor = [100, 100]):

        # Params
        self.N_agents  = N_agents
        self.state_dim = state_dim
        self.act_dim   = act_dim
        self.gamma     = gamma
        self.tau       = tau

        # Critics
        self.critic        = Critic(N_agents, state_dim, act_dim, hidden_size_critic)
        self.critic_target = Critic(N_agents, state_dim, act_dim, hidden_size_critic)

        # Actors
        self.actors = [Actor(state_dim, act_dim, hidden_size_actor) for i in range(N_agents)]
        self.actors_target = [Actor(state_dim, act_dim, hidden_size_actor) for i in range(N_agents)]

        # Initialize weights
        # Critic
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Actors
        for actor, actor_target in zip(self.actors, self.actors_target):
            for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                target_param.data.copy_(param.data)

        # Replay Buffer


if __name__ == '__main__':

    state = torch.tensor(np.random.rand(12,))
    action = torch.tensor(np.random.rand(6,))
    #state = torch.from_numpy(numpy_trash).double()
    test_MADDPG = MADDPG(3, 4, 2)
    ipdb.set_trace()

