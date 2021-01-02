from Actor import Actor
from Critic import Critic
import ReplayBuffer
from ConstraintNetwork import ConstraintNetwork

import torch
import torch.autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

import numpy as np
import ipdb


import cvxpy
import osqp
import scipy
from scipy import sparse


# Define Solver Globaly
solver = osqp.OSQP()

class SafeDDPGagent:

    def __init__(self, state_dim, act_dim, constraint_dim, num_agents, col_margin = 0.5 ,hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=6000):
        # Params
        self.state_dim  = state_dim
        self.act_dim    = act_dim
        self.constraint_dim = constraint_dim
        self.num_agents = num_agents
        self.gamma      = gamma
        self.tau        = tau
        self.col_margin = col_margin


        # Import Constraint Networks
        self.constraint_net_1 = ConstraintNetwork(self.state_dim*self.num_agents, self.act_dim*self.num_agents)
        self.constraint_net_2 = ConstraintNetwork(self.state_dim*self.num_agents, self.act_dim*self.num_agents)
        self.constraint_net_3 = ConstraintNetwork(self.state_dim*self.num_agents, self.act_dim*self.num_agents)
        self.constraint_net_4 = ConstraintNetwork(self.state_dim*self.num_agents, self.act_dim*self.num_agents)
        self.constraint_net_5 = ConstraintNetwork(self.state_dim*self.num_agents, self.act_dim*self.num_agents)
        self.constraint_net_6 = ConstraintNetwork(self.state_dim*self.num_agents, self.act_dim*self.num_agents)

        self.constraint_net_1.load_state_dict(torch.load("constraint_net1.pkl"))
        self.constraint_net_2.load_state_dict(torch.load("constraint_net2.pkl"))
        self.constraint_net_3.load_state_dict(torch.load("constraint_net3.pkl"))
        self.constraint_net_4.load_state_dict(torch.load("constraint_net4.pkl"))
        self.constraint_net_5.load_state_dict(torch.load("constraint_net5.pkl"))
        self.constraint_net_6.load_state_dict(torch.load("constraint_net6.pkl"))

        self.constraint_nets = [self.constraint_net_1, \
                                self.constraint_net_2, \
                                self.constraint_net_3, \
                                self.constraint_net_4, \
                                self.constraint_net_5, \
                                self.constraint_net_6]


        # Initiallize OSQP solver
        self.init_osqp()


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



    def init_osqp(self):

        global solver

        # (2) Problem Variables
        # Problem specific constants
        I    = np.eye(self.act_dim * self.num_agents)
        ones = np.ones(self.act_dim * self.num_agents)
        C    = np.zeros(self.constraint_dim * self.num_agents)

        # Formulate the constraints using neural networks
        G    = np.zeros([self.act_dim * self.num_agents, self.act_dim * self.num_agents])

        # (2) Problem Variables in QP form

        #q = scipy.sparse.csc_matrix(-actions)
        q = -np.zeros(self.act_dim * self.num_agents)
        P = scipy.sparse.eye(self.act_dim * self.num_agents)

        A = scipy.sparse.csc_matrix(np.concatenate([-G, I, -I]))
        u = np.concatenate([C, ones, ones])
        l = None

        # (2) Update Solver
        solver.setup(P, q, A, l, u,verbose=False)

    @torch.no_grad()
    def get_action(self, state, constraint):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor(state)
        action = action.detach().numpy().flatten()

        # transform numpy array into list of 3 actions
        actions = self.correct_actions(state, action, constraint)

        actions = np.split(action, self.num_agents)
        return actions

    @torch.no_grad()
    def correct_actions_old(self, state, actions, constraint):

        # QP solution should be here
        x = cvxpy.Variable(self.act_dim * self.num_agents)

        # Define the cost function
        cost_fun = cvxpy.norm(actions - x)**2

        # Define the Constraints
        # C + g * a < 
        C = np.concatenate(constraint)

        # Formulate the constraints using neural networks
        A = np.zeros([self.act_dim * self.num_agents, self.act_dim * self.num_agents])

        for i, net in enumerate(self.constraint_nets):
            A[i, :] = net(state).numpy()

        # Box constraints
        I    = np.eye(self.act_dim * self.num_agents)
        ones = np.ones(self.act_dim * self.num_agents)
        constr = [-A @ x <= C - self.col_margin, I @ x <= ones, -I @ x <= ones]

        # Define Optimization Problem
        prob = cvxpy.Problem(cvxpy.Minimize(cost_fun), constr)
        prob.solve()

        return x.value

    @torch.no_grad()
    def correct_actions(self, state, actions, constraint):

        # (1) Create solver as a globar variable
        global solver
        #ipdb.set_trace()


        # (2) Problem Variables
        # Problem specific constants
        I    = np.eye(self.act_dim * self.num_agents)
        ones = np.ones(self.act_dim * self.num_agents)
        C    = np.concatenate(constraint)

        # Formulate the constraints using neural networks
        G    = np.zeros([self.act_dim * self.num_agents, self.act_dim * self.num_agents])
        for i, net in enumerate(self.constraint_nets):
            G[i, :] = net(state).numpy()

        # (2) Problem Variables in QP form
        '''
        q_ = -actions
        P_ = np.eye(self.act_dim * self.num_agents)

        A_ = np.concatenate([-G, I, -I])
        u_ = np.concatenate([C - self.col_margin, ones, ones])
        l = None
        '''

        #q = scipy.sparse.csc_matrix(-actions)
        q = -actions
        P = scipy.sparse.eye(self.act_dim * self.num_agents)

        A = scipy.sparse.csc_matrix(np.concatenate([-G, I, -I]))
        u = np.concatenate([C - self.col_margin, ones, ones])
        l = None

        # (2) Update Solver
        solver.update(q = q, l = l, u = u, Ax = A.data)
        x = solver.solve()
        return x.x

        # QP solution should be here
        x = cvxpy.Variable(self.act_dim * self.num_agents)

        # Define the cost function
        cost_fun = cvxpy.norm(actions - x)**2

        # Define the Constraints
        # C + g * a < 
        C = np.concatenate(constraint)

        # Formulate the constraints using neural networks
        A = np.zeros([self.act_dim * self.num_agents, self.act_dim * self.num_agents])

        for i, net in enumerate(self.constraint_nets):
            A[i, :] = net(state).numpy()

        # Box constraints
        I    = np.eye(self.act_dim * self.num_agents)
        ones = np.ones(self.act_dim * self.num_agents)
        constr = [-A @ x <= C - self.col_margin, I @ x <= ones, -I @ x <= ones]

        # Define Optimization Problem
        prob = cvxpy.Problem(cvxpy.Minimize(cost_fun), constr)
        prob.solve()

        return x.value

    '''
    def constraint(self,state, margin = self.col_margin, mode = 1):
        if (mode == 1):
            constraint_val = -np.sum(np.multiply(state[-2:], state[-2:])) - margin**2
        elif (mode == 2):
            constraint_val = -np.sum(np.multiply(state[-4:-2], state[-4:-2])) - margin**2
        return constraint_val
    '''

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

