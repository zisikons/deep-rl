import numpy as np
class OUNoise(object):
    def __init__(self, act_dim, num_agents, act_low, act_high, mu=0.0,
            theta=0.15, max_sigma=0.7, min_sigma=0.05, decay_period=2500):
        # Parameters
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.num_agents   = num_agents
        self.action_dim   = act_dim
        self.low          = act_low
        self.high         = act_high
        self.reset()

    def reset(self):
        self.state = np.ones(self.num_agents*self.action_dim) * self.mu

    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + \
             self.sigma * np.random.randn(self.num_agents*self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t, episode):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - \
                    (self.max_sigma - self.min_sigma) * (episode / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
