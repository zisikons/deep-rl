from Actor import Actor
from Critic import Critic
import ReplayBuffer

import torch
import torch.autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

import numpy as np
import ipdb
class DDPGagent:
     
    def __init__(self, state_dim, act_dim, num_agents ,hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=6000):
        # Params
        self.state_dim  = state_dim
        self.act_dim    = act_dim
        self.num_agents = num_agents
        self.gamma      = gamma
        self.tau        = tau
        
        # Networks
        self.actor = Actor(self.state_dim, self.act_dim, self.num_agents)
        self.actor_target = Actor(self.state_dim, self.act_dim, self.num_agents)
        self.critic = Critic(self.state_dim, self.act_dim, self.num_agents)
        self.critic_target = Critic(self.state_dim, self.act_dim, self.num_agents)
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)


        # Training
        self.memory = ReplayBuffer.ReplayBuffer(self.state_dim, self.act_dim, self.num_agents, max_memory_size)
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
        
        #with torch.no_grad():
        next_actions = self.actor_target(next_states)
        next_Q = self.critic_target(next_states, next_actions.detach()).squeeze(1)
        Qprime = rewards + self.gamma * next_Q
        assert Qprime.requires_grad == True
       
        # Standerdize the data 
        #Qprime = (Qprime - torch.mean(Qprime).detach())/torch.std(Qprime.detach())
        #Qval   = (Qvals - torch.mean(Qvals.detach()))/torch.std(Qvals.detach())
 
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

