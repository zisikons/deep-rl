import torch
import torch.autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

import numpy as np
import scipy as sp

from qpsolvers import solve_qp

from core.DDPG import DDPGagent
from core.constraint_network import ConstraintNetwork


class SafeDDPGagent(DDPGagent):
    def __init__(self, state_dim, act_dim, constraint_dim, num_agents, constraint_networks_dir, col_margin = 0.35 ,
            hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3,
            gamma=0.99, tau=1e-2, max_memory_size=16000,  soften = True):

        # Call DDPGagent's constructor
        super().__init__(state_dim, act_dim, num_agents ,hidden_size,
                         actor_learning_rate, critic_learning_rate, gamma,
                         tau, max_memory_size)

        # Extra Params
        self.col_margin = col_margin
        self.constraint_dim = constraint_dim

        self.total_state_dim = self.state_dim * self.num_agents
        self.total_constraint_dim = self.constraint_dim * self.num_agents
        self.total_action_dim = self.act_dim * self.num_agents

        self.constraint_nets = self.total_constraint_dim*[None]
        
        # Initialize constraint networks
        for i in range(self.total_constraint_dim):
            self.constraint_nets[i] = ConstraintNetwork(self.total_state_dim, self.total_action_dim)
            self.constraint_nets[i].load_state_dict(torch.load(constraint_networks_dir
                                                    + "constraint_net_" + str(i) + ".pkl"))

        # Define Solver Globaly
        self.solver_interventions = 0
        self.solver_infeasible    = 0

        # Choose Solver
        if soften:
            self.correct_actions = self.correct_actions_soften
        else:
            self.correct_actions = self.correct_actions_hard


    def reset_metrics(self):
        self.solver_interventions = 0
        self.solver_infeasible    = 0

    def get_interventions(self):
        return self.solver_interventions

    def get_infeasible(self):
        return self.solver_infeasible

    @torch.no_grad()
    def get_action(self, state, constraint):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor(state)
        action = action.detach().numpy().flatten()

        # transform numpy array into list of 3 actions
        action = self.correct_actions(state, action, constraint)

        actions = np.split(action, self.num_agents)
        return actions

    @torch.no_grad()
    def correct_actions_hard(self, state, actions, constraint):

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
    def correct_actions_soften(self, state, actions, constraint):

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
        P = sp.linalg.block_diag(I, Z + I * 0.001, Z + I * 0.001)
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
        if np.linalg.norm(actions - x) > 1e-3:
            self.solver_interventions += 1

        return x
