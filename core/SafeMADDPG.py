import torch
import torch.autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

import numpy as np
import scipy as sp
import scipy.linalg

from qpsolvers import solve_qp

from core.MADDPG import MADDPGagent
from core.constraint_network import ConstraintNetwork
import ipdb


class SafeMADDPGagent(MADDPGagent):
    #def __init__(self, state_dim, act_dim, constraint_dim, num_agents, constraint_networks_dir, col_margin = 0.35 ,
    #        hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3,
    #        gamma=0.99, tau=1e-2, max_memory_size=16000,  soften = True):

    def __init__(self, N_agents, state_dim, act_dim,
                 constraint_networks_dir, constraint_dim,critic_state_mask = [0,1,2,3,-1,-2], col_margin=0.35,
                 actor_learning_rate=1e-4,
                 critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=30000,
                 hidden_size_critic = [500, 500], hidden_size_actor = [100, 100],
                 batch_size = 128, soften = True):

        # Call MADDPGagent's constructor
        super().__init__(N_agents = N_agents, state_dim = state_dim, 
                         act_dim = act_dim, critic_state_mask = critic_state_mask, 
                         actor_learning_rate = actor_learning_rate,
                         critic_learning_rate = critic_learning_rate, gamma = gamma,
                         tau = tau, max_memory_size = max_memory_size,
                         hidden_size_critic = hidden_size_critic, hidden_size_actor = hidden_size_actor,
                         batch_size = batch_size)

        # Extra Params
        self.col_margin = col_margin
        self.constraint_dim = constraint_dim

        self.total_state_dim = self.state_dim * self.N_agents
        self.total_constraint_dim = self.constraint_dim * self.N_agents
        self.total_action_dim = self.act_dim * self.N_agents

        self.constraint_nets = self.total_constraint_dim*[None]

        # Initialize constraint networks
        for i in range(self.total_constraint_dim):
            self.constraint_nets[i] = ConstraintNetwork(self.total_state_dim, self.total_action_dim).double()
            self.constraint_nets[i].load_state_dict(torch.load(constraint_networks_dir
                                                    + "constraint_net_" + str(i) + ".pkl"))

        # Define Solver Globaly
        self.solver_interventions = 0
        self.solver_infeasible    = 0

        # Choose Solver
        if soften:
            self.correct_actions = self.correct_actions_soften2
        else:
            self.correct_actions = self.correct_actions_hard

        self.soften = soften

    def reset_metrics(self):
        self.solver_interventions = 0
        self.solver_infeasible    = 0

    def get_interventions(self):
        return self.solver_interventions

    def get_infeasible(self):
        return self.solver_infeasible

    @torch.no_grad()
    def get_action(self, state, constraint):

        # Original MADDPG
        actions = []
        for i in range(self.N_agents):
            s = torch.tensor(state[i], dtype=torch.float64)
            action = self.actors[i](s).detach()
            actions.append(action)

        # merge action and state vectors of all agents
        action_total = torch.cat(actions)
        state_total  = torch.tensor(np.concatenate(state),dtype=torch.float64)

        # correct unsafe actions
        if (self.soften):
            action, intervention_metric = self.correct_actions(state_total, action_total, constraint)
            
            # transform numpy array into list of 3 actions
            actions = np.split(action, self.N_agents)
            return actions, intervention_metric

        else:
            action = self.correct_actions(state_total,action_total, constraint)
            
            # transform numpy array into list of 3 actions
            actions = np.split(action, self.N_agents)
            return actions

    @torch.no_grad()
    def get_action2(self, state, constraint):

        # Original MADDPG
        actions = []
        for i in range(self.N_agents):
            s = torch.tensor(state[i], dtype=torch.float64)
            action = self.actors[i](s).detach()
            actions.append(action)
        # merge action and state vectors of all agents
        action_total = torch.cat(actions).numpy()
        return actions

    @torch.no_grad()
    def correct_actions_hard(self, state, actions, constraint):

        actions = actions.numpy()
        # (1) Problem Variables
        # Problem specific constants
        I    = np.eye(self.total_action_dim)
        ones = np.ones(self.total_action_dim)
        C    = np.concatenate(constraint)

        # Formulate the constraints using neural networks
        G    = np.zeros([self.total_action_dim, self.total_action_dim])
        for i, net in enumerate(self.constraint_nets):
            G[i, :] = net(state).numpy()

        # (2) Problem Variables in QP form
        # Cost Function
        q = -actions
        P = np.eye(self.total_action_dim)

        # Constraints
        A = np.concatenate([-G, I, -I])
        ub = np.concatenate([C - self.col_margin, ones, ones])
        lb = None

        # Solve Optimization Problem
        try:
            x = solve_qp(P.astype(np.float64), q.astype(np.float64), A.astype(np.float64),
                         ub.astype(np.float64), None, None, None, None)
        except:
            self.solver_infeasible +=1
            return actions

        # Count Solver interventions
        if np.linalg.norm(actions - x) > 1e-3:
            self.solver_interventions += 1

        return x

    @torch.no_grad()
    def correct_actions_hard2(self, state, actions, constraint):

        actions = np.concatenate(actions)
        state = torch.tensor(np.concatenate(state))

        # (1) Problem Variables
        # Problem specific constants
        I    = np.eye(self.total_action_dim)
        ones = np.ones(self.total_action_dim)
        C    = np.concatenate(constraint)

        # Formulate the constraints using neural networks
        G    = np.zeros([self.total_action_dim, self.total_action_dim])
        for i, net in enumerate(self.constraint_nets):
            G[i, :] = net(state).numpy()

        # (2) Problem Variables in QP form
        # Cost Function
        q = -actions
        P = np.eye(self.total_action_dim)

        # Constraints
        A = np.concatenate([-G, I, -I])
        ub = np.concatenate([C - self.col_margin, ones, ones])
        lb = None

        # Solve Optimization Problem
        try:
            x = solve_qp(P.astype(np.float64), q.astype(np.float64), A.astype(np.float64),
                         ub.astype(np.float64), None, None, None, None)
        except:
            self.solver_infeasible +=1
            return actions

        # Count Solver interventions
        if np.linalg.norm(actions - x) > 1e-3:
            self.solver_interventions += 1

        return x

    @torch.no_grad()
    def correct_actions_soften2(self, state, actions, constraint):

        actions = np.concatenate(actions)
        state = torch.tensor(np.concatenate(state))
        # (1) Create solver as a globar variable
        l1_penalty = 1000

        # (2) Problem Variables
        # Problem specific constants
        I     = np.eye(self.total_action_dim)
        Z     = np.zeros([self.total_action_dim, self.total_action_dim])
        ones  = np.ones(self.total_action_dim)
        zeros = np.zeros(self.total_action_dim)
        C     = np.concatenate(constraint) - self.col_margin

        # Formulate the constraints using neural networks
        G    = np.zeros([self.total_action_dim, self.total_action_dim])
        for i, net in enumerate(self.constraint_nets):
            G[i, :] = net(state).numpy()

        # (2) Problem Variables in QP form
        # Cost Function
        P = sp.linalg.block_diag(I, Z + I * 0.000001, Z + I * 0.000001)
        q = np.concatenate([-actions, ones, zeros])

        # Constraints
        A = np.vstack((np.concatenate([-G, Z, -I], axis = 1),
                       np.concatenate([Z, Z, -I], axis = 1),
                       np.concatenate([Z, -I,  l1_penalty * I], axis = 1),
                       np.concatenate([Z, -I, -l1_penalty * I], axis = 1)))

        ub = np.concatenate((C, zeros, zeros, zeros))
        lb = None

        # Solve Optimization Problem
        try:
            x = solve_qp(P.astype(np.float64), q.astype(np.float64), A.astype(np.float64),
                         ub.astype(np.float64), None, None, None, None)
            x = x[0:(self.total_action_dim)]
        except:
            self.solver_infeasible +=1
            return actions

        # Count Solver interventions
        norm_diff = np.linalg.norm(actions-x)
        if norm_diff > 1e-3:
            self.solver_interventions += 1

        # calculating an intervetion metric 
        intervention_metric = np.split(np.abs(actions - x), self.N_agents)
        intervention_metric = [np.sum(i) for i in intervention_metric]
        return x, intervention_metric

    @torch.no_grad()
    def correct_actions_soften(self, state, actions, constraint):

        actions = actions.numpy()
        # (1) Create solver as a globar variable
        l1_penalty = 1000

        # (2) Problem Variables
        # Problem specific constants
        I     = np.eye(self.total_action_dim)
        Z     = np.zeros([self.total_action_dim, self.total_action_dim])
        ones  = np.ones(self.total_action_dim)
        zeros = np.zeros(self.total_action_dim)
        C     = np.concatenate(constraint) - self.col_margin

        # Formulate the constraints using neural networks
        G    = np.zeros([self.total_action_dim, self.total_action_dim])
        for i, net in enumerate(self.constraint_nets):
            G[i, :] = net(state).numpy()

        # (2) Problem Variables in QP form
        # Cost Function
        P = sp.linalg.block_diag(I, Z + I * 0.000001, Z + I * 0.000001)
        q = np.concatenate([-actions, ones, zeros])

        # Constraints
        A = np.vstack((np.concatenate([-G, Z, -I], axis = 1),
                       np.concatenate([Z, Z, -I], axis = 1),
                       np.concatenate([Z, -I,  l1_penalty * I], axis = 1),
                       np.concatenate([Z, -I, -l1_penalty * I], axis = 1)))

        ub = np.concatenate((C, zeros, zeros, zeros))
        lb = None

        # Solve Optimization Problem
        try:
            x = solve_qp(P.astype(np.float64), q.astype(np.float64), A.astype(np.float64),
                         ub.astype(np.float64), None, None, None, None)
            x = x[0:(self.total_action_dim)] 
        except:
            self.solver_infeasible +=1
            return actions

        # Count Solver interventions
        norm_diff = np.linalg.norm(actions-x)
        if norm_diff > 1e-3:
            self.solver_interventions += 1
        
        # calculating an intervetion metric 
        intervention_metric = np.split(np.abs(actions - x), self.N_agents)
        intervention_metric = [np.sum(i) for i in intervention_metric]
        return x, intervention_metric 
