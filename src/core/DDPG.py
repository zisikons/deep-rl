import os
import numpy as np
import ipdb

import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn

from torch.autograd import Variable

def combined_shape(length, shape=None):
    """
    Helper function that combines two array shapes.
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

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



class Critic(nn.Module):
    """
    Critic neural network
    """
    def __init__(self, state_dim = 12, act_dim = 2, hidden_size = [500, 500]):
        """
        Constructor
        Arguments:
            - state_dim   : the state dimension of the RL agent
            - action_dim  : the action dimension of the RL agent
            - hidden_size : hidden layer size
        """
        super(Critic, self).__init__()

        # Dimensions
        self.input_size  = state_dim + act_dim
        self.hidden_size = hidden_size
        self.output_size = 1

        # Network Architecture
        self.linear_1 = nn.Linear(self.input_size, self.hidden_size[0])
        self.linear_2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.linear_3 = nn.Linear(self.hidden_size[1], self.output_size)
        layers = [self.linear_1, self.linear_2, self.linear_3]

        # initialize the weights
        for layer in layers:
            torch.nn.init.xavier_uniform_(layer.weight)


    def forward(self, state, action):
        """
        Forward propagation method
        """
        # Params state and actions are torch tensors
        x = torch.cat([state, action], 1)
        x = nn.ReLU()(self.linear_1(x))
        x = nn.ReLU()(self.linear_2(x))
        x = self.linear_3(x)
        return x


class Actor(nn.Module):
    def __init__(self, state_dim=12, act_dim=2,  hidden_size = [100, 100]):
        super(Actor, self).__init__()

        self.input_size  = state_dim
        self.output_size = act_dim
        self.hidden_size = hidden_size

        # Network Architecture
        self.linear_1 = nn.Linear(self.input_size, self.hidden_size[0])
        self.linear_2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.linear_3 = nn.Linear(self.hidden_size[1], self.output_size)

        layers = [self.linear_1, self.linear_2, self.linear_3]

        # initialize the weights
        for layer in layers:
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = nn.ReLU()(self.linear_1(state))
        x = nn.ReLU()(self.linear_2(x))
        x = nn.Tanh()(self.linear_3(x))
        return x


class DDPGagent:

    def __init__(self, state_dim, act_dim, num_agents ,hidden_size=256, actor_learning_rate=1e-4,
                 critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=16000):
        # Params
        self.state_dim  = state_dim
        self.act_dim    = act_dim
        self.num_agents = num_agents
        self.gamma      = gamma
        self.tau        = tau

        # Network Dimensions
        NN_state_dim = state_dim * num_agents
        NN_act_dim = act_dim * num_agents

        # Networks
        self.actor = Actor(NN_state_dim, NN_act_dim)
        self.actor_target = Actor(NN_state_dim, NN_act_dim)
        self.critic = Critic(NN_state_dim, NN_act_dim)
        self.critic_target = Critic(NN_state_dim, NN_act_dim)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)


        # Training
        self.memory = ReplayBuffer(self.state_dim, self.act_dim, self.num_agents, max_memory_size)
        self.critic_criterion  = nn.MSELoss(reduction = 'mean')
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    @torch.no_grad()
    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor(state)
        action = action.detach().numpy().flatten()
        # transform numpy array into list of 3 actions
        actions = np.split(action, self.num_agents)
        return actions


    # redundant
    def get_data(self):
        return self.memory.get()

    def update(self,data, batch_size):

        data_size = data['state'].shape[0]
        # Sample the data
        sample_idx = np.random.choice(np.arange(data_size), batch_size)

        states = data['state'][sample_idx,:]
        actions = data['act'][sample_idx,:]
        rewards = data['rew'][sample_idx]
        next_states = data['next_state'][sample_idx,:]

        # Critic loss        
        Qvals = self.critic(states, actions).squeeze(1)
        assert Qvals.requires_grad == True

        next_actions = self.actor_target(next_states)
        next_Q = self.critic_target(next_states, next_actions.detach()).squeeze(1)
        Qprime = rewards + self.gamma * next_Q

        assert Qprime.requires_grad == True
        assert Qprime.shape[0] == batch_size

        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        a = self.critic(states, self.actor(states))

        policy_loss = -self.critic(states, self.actor(states)).mean()
        assert policy_loss.requires_grad == True

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


    def save_params(self, directory):

        # Export Results for training
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save Actor Network
        torch.save(self.actor.state_dict(), directory + 'actor_net.pkl')

        # Save Critic Network
        torch.save(self.critic.state_dict(), directory + 'critic_net.pkl')

    def load_params(self, directory):

        # Export Results for training
        if not os.path.exists(directory):
            raise Exception('There exists no such directory.')

        self.actor.load_state_dict(torch.load(directory + "actor_net.pkl"))
        self.critic.load_state_dict(torch.load(directory + "critic_net.pkl"))
