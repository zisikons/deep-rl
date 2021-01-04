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

from qpsolvers import solve_qp

import scipy as sp
import scipy.linalg

# Define Solver Globaly
solver = osqp.OSQP()
solver_interventions = 0

def get_interventions():
    global solver_interventions
    return solver_interventions

class SafeDDPGagent:

    def __init__(self, state_dim, act_dim, constraint_dim, num_agents, col_margin = 0.35 ,hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=6000):
        # Params
        self.state_dim  = state_dim
        self.act_dim    = act_dim
        self.constraint_dim = constraint_dim
        self.num_agents = num_agents
        self.gamma      = gamma
        self.tau        = tau
        self.col_margin = col_margin

        # Define Solver Globaly
        self.solver = osqp.OSQP()
        self.solver_interventions = 0
        self.solver_infeasible    = 0


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
        self.reset_metrics()


        # Choose Solver
        #self.correct_actions = self.correct_actions_cvxpy
        self.correct_actions = self.correct_actions_osqp

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


    def reset_metrics(self):
        self.solver_interventions = 0
        self.solver_infeasible    = 0

    def get_interventions(self):
        return self.solver_interventions

    def get_infeasible(self):
        return self.solver_infeasible

    def init_osqp(self):

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
        self.solver.setup(P, q, A, l, u,verbose=False, eps_prim_inf = 1e-04)

    @torch.no_grad()
    def get_action(self, state, constraint):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor(state)
        action = action.detach().numpy().flatten()

        # transform numpy array into list of 3 actions
        action_qprog = self.correct_actions_soften(state, action, constraint)
        #action_cvx = self.correct_actions_cvxpy(state, action, constraint)
        #action_osqp = self.correct_actions_osqp(state, action, constraint)

        #if np.linalg.norm(action_cvx - action_qprog) > 0.2:
        #    ipdb.set_trace()

        action = action_qprog

        actions = np.split(action, self.num_agents)
        return actions

    def predict_constraints(self, state):
        # Formulate the constraints using neural networks
        A = np.zeros([self.act_dim * self.num_agents, self.act_dim * self.num_agents])

        for i, net in enumerate(self.constraint_nets):
            A[i, :] = net(state).numpy()

        return A

    @torch.no_grad()
    def correct_actions_cvxpy(self, state, actions, constraint):

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

        if prob.status == 'infeasible' or prob.status == 'infeasible_inaccurate':
            self.solver_infeasible += 1
            #print('cvxpy infeasible')
            return actions

        if np.linalg.norm(x.value - actions) >1e-3:
            self.solver_interventions +=1

        return x.value

    @torch.no_grad()
    def correct_actions_osqp(self, state, actions, constraint):

        # (1) Create solver as a globar variable
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
        self.solver.update(q = q, l = l, u = u, Ax = A.data)
        x = self.solver.solve()

        if any(x.x == None):
            # print("Houston, we have problem")
            self.solver_infeasible +=1
            #print('OSQP infeasible')
            return actions

        if np.linalg.norm(actions - x.x) > 1e-3:
            self.solver_interventions += 1

        return x.x

    @torch.no_grad()
    def correct_actions_quadprog(self, state, actions, constraint):

        # (1) Create solver as a globar variable
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
        P = np.eye(self.act_dim * self.num_agents)

        A = np.concatenate([-G, I, -I])
        ub = np.concatenate([C - self.col_margin, ones, ones])

        lb = None

        try:
            x = solve_qp(P.astype(np.float64), q.astype(np.float64), A.astype(np.float64), ub.astype(np.float64), None, None, None, None)

        except:
            # print("Houston, we have problem")
            self.solver_infeasible +=1
            #print('QUADPROG infeasible')
            #ipdb.set_trace()
            return actions

        if np.linalg.norm(actions - x) > 1e-3:
            self.solver_interventions += 1

        return x

    def correct_actions_soften(self, state, actions, constraint):

        # (1) Create solver as a globar variable
        #ipdb.set_trace()
        l1_penalty = 1000

        # (2) Problem Variables
        # Problem specific constants
        I     = np.eye(self.act_dim * self.num_agents)
        Z     = np.zeros([self.act_dim * self.num_agents,self.act_dim * self.num_agents])
        ones  = np.ones(self.act_dim * self.num_agents)
        zeros = np.zeros(self.act_dim * self.num_agents)
        C     = np.concatenate(constraint) - self.col_margin

        # Formulate the constraints using neural networks
        G    = np.zeros([self.act_dim * self.num_agents, self.act_dim * self.num_agents])
        for i, net in enumerate(self.constraint_nets):
            G[i, :] = net(state).numpy()

        # (2) Problem Variables in QP form
        P = sp.linalg.block_diag(I, Z + I * 0.001, Z + I * 0.001)
        q = np.concatenate([-actions, ones, zeros])

        A = np.vstack((np.concatenate([-G, Z, -I], axis = 1),
                       np.concatenate([Z, Z, -I], axis = 1),
                       np.concatenate([Z, -I,  l1_penalty * I], axis = 1),
                       np.concatenate([Z, -I, -l1_penalty * I], axis = 1)))

        ub = np.concatenate((C, zeros, zeros, zeros))
        lb = None

        try:
            x = solve_qp(P.astype(np.float64), q.astype(np.float64), A.astype(np.float64), ub.astype(np.float64), None, None, None, None)

        except:
            # print("Houston, we have problem")
            self.solver_infeasible +=1
            #print('QUADPROG infeasible')
            #ipdb.set_trace()
            return actions

        x = x[0:(self.act_dim * self.num_agents)]

        if np.linalg.norm(actions - x) > 1e-3:
            self.solver_interventions += 1

        return x
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

