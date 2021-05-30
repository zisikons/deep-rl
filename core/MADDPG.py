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
    def __init__(self, size):

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
        # Old version
        #self.state_buf.append(np.expand_dims(state, axis = 0))
        #self.act_buf.append(np.expand_dims(act, axis = 0))
        #self.rew_buf.append(np.array(rew, ndmin = 1))
        #self.next_state_buf.append(np.expand_dims(next_state, axis = 0))

        # New version (best suited for decentralized)
        self.state_buf.append(state)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.next_state_buf.append(next_state)
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
    def __init__(self, state_size, action_size, hidden_size = [500, 500]):
        """
        Constructor
        Arguments:
            - state_dim   : the state dimension of the critic network state
            - action_size : the total action dimension of the critic network action
            - hidden_size : hidden layer size
        """
        super(Critic, self).__init__()

        # Dimensions
        self.input_size  = state_size + action_size
        self.hidden_size = hidden_size
        self.output_size = 1

        # Network Architecture
        self.layers = nn.ModuleList()

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
        x = torch.cat([state, action], 1) #2nd index = dim of concat

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
        self.layers = nn.ModuleList()

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

# Note: Buffer size is changed
class MADDPGagent:

    def __init__(self, N_agents, state_dim, act_dim, actor_learning_rate=1e-4,
                 critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=30000,
                 hidden_size_critic = [500, 500], hidden_size_actor = [100, 100],
                 batch_size = 128):

        # Params
        self.N_agents  = N_agents
        self.state_dim = state_dim
        self.act_dim   = act_dim
        self.gamma     = gamma
        self.tau       = tau
        self.batch_size = batch_size

        # Critics
        # Note: Sunday quick and dirty hack to avoid duplicate states
        self.critic_state_mask = [0, 1, 2, 3, 8, 9]
        self.critic_state_dim = N_agents * len(self.critic_state_mask)
        self.critic_act_dim   = N_agents * self.act_dim

        self.critics = [Critic(self.critic_state_dim,
                               self.critic_act_dim,
                               hidden_size_critic) for i in range(N_agents)]

        self.critics_target = [Critic(self.critic_state_dim,
                                      self.critic_act_dim,
                                      hidden_size_critic) for i in range(N_agents)]

        # Actors
        self.actors = [Actor(state_dim, act_dim, hidden_size_actor) for i in range(N_agents)]
        self.actors_target = [Actor(state_dim, act_dim, hidden_size_actor) for i in range(N_agents)]

        # Initialize weights
        # Critic
        for critic, critic_target in zip(self.critics, self.critics_target):
            for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                target_param.data.copy_(param.data)

        # Actors
        for actor, actor_target in zip(self.actors, self.actors_target):
            for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                target_param.data.copy_(param.data)

        # Replay Buffer
        self.memory = ReplayBuffer(max_memory_size)


        # Loss functions and other weird artifacts
        self.critic_criterion  = nn.MSELoss(reduction = 'mean')

        #yolo_optimizer = optim.Adam(self.actors[1].parameters, lr = actor_learning_rate)
        #ipdb.set_trace()
        self.actor_optimizers = [optim.Adam(self.actors[i].parameters(), lr=actor_learning_rate)
                                 for i in range(N_agents)]
        self.critic_optimizers = [optim.Adam(self.critics[i].parameters(), lr=critic_learning_rate)
                                  for i in range(N_agents)]



    @torch.no_grad()
    def get_action(self, state):

        actions = []
        for i in range(self.N_agents):
            s = torch.tensor(state[i], dtype=torch.float64)
            action = self.actors[i](s).detach().numpy().flatten()
            actions.append(action)

        return actions

    def update(self):

        # Sample a batch from replay buffer
        data_size = self.memory.ptr
        sample_idx = np.random.choice(np.arange(data_size), self.batch_size)

        states      = [self.memory.state_buf[i] for i in sample_idx]
        actions     = [self.memory.act_buf[i] for i in sample_idx]
        rewards     = [self.memory.rew_buf[i] for i in sample_idx]
        next_states = [self.memory.next_state_buf[i] for i in sample_idx]

        # Convert to "correct" input format for Pytorch NNs
        batch = {'states' : states,
                 'actions' : actions,
                 'rewards' : rewards,
                 'next_states' : next_states}

        # Transform data from Replay Buffer to network inputs
        x_crit, a_crit = self.get_Q_state(batch)
        reward_mat = np.array(rewards)

        critics_loss  = []
        policy_losses = []
        for idx, (actor, critic, actor_target, critic_target) in enumerate(zip(self.actors,
                                                                               self.critics,
                                                                            self.actors_target,
                                                                            self.critics_target)):

            # Q-values for current state
            Q_vals = critic(x_crit, a_crit).squeeze(1)

            # Evaluate next state
            next_actions   = []
            next_state_mat = np.array(next_states)

            for idx_, actor_ in enumerate(self.actors_target):
                #print(f'idx_ = {idx_}')
                next_actions.append(
                        actor_(torch.tensor(next_state_mat[:,idx_,:],dtype=torch.float64)))
            # merge next actions
            A_prime = torch.cat(next_actions, axis = 1)

            # Q-values for next state
            next_state_mat = next_state_mat[:, :, self.critic_state_mask]
            S_prime = torch.tensor(next_state_mat.reshape(self.batch_size, self.critic_state_dim),
                                                        dtype=torch.float64)

            next_Q = critic_target(S_prime, A_prime).squeeze(1)

            # Copmute Q_prime and loss based on the reward
            Q_prime = torch.tensor(reward_mat[:, idx], dtype=torch.float64) + self.gamma * next_Q
            critics_loss.append(self.critic_criterion(Q_vals, Q_prime))

            # Update Actor
            policy_losses.append(-critic(x_crit, a_crit).mean())

        for i in range(self.N_agents):
            self.actor_optimizers[i].zero_grad()
            policy_losses[i].backward()
            self.actor_optimizers[i].step()

            self.critic_optimizers[i].zero_grad()
            critics_loss[i].backward()
            self.critic_optimizers[i].step()

            # update target networks 
            for target_param, param in zip(self.actors_target[i].parameters(), self.actors[i].parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.critics_target[i].parameters(), self.critics[i].parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def get_Q_state(self, batch):

        x_input = torch.zeros((self.batch_size, self.critic_state_dim), dtype=torch.float64)
        a_input = torch.zeros((self.batch_size, self.critic_act_dim), dtype=torch.float64)

        for t in range(self.batch_size):
            # Get states
            x = []
            desired_idx = [0, 1, 2, 3, 8, 9]
            for agent in range(self.N_agents):
                x += [torch.tensor(batch['states'][t][agent][desired_idx], dtype=torch.float64)]

            x_input[t, :] = torch.cat(x)

            # Get actions
            a = []
            for agent in range(self.N_agents):
                a += [torch.tensor(batch['actions'][t][agent], dtype=torch.float64)]

            a_input[t, :] = torch.cat(a)

        # Convert to torch tensors
        #x_input = torch.tensor(x_input)
        #a_input = torch.tensor(a_input)

        return x_input, a_input

if __name__ == '__main__':

    state = torch.tensor(np.random.rand(12,))
    action = torch.tensor(np.random.rand(6,))
    #state = torch.from_numpy(numpy_trash).double()
    test_MADDPG = MADDPG(3, 4, 2)
    ipdb.set_trace()

