import torch
import numpy as np

def combined_shape(length, shape=None):
    """Helper function that combines two array shapes."""
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class ReplayBuffer:
    """
    Buffer to store trajectories.
    """
    def __init__(self, state_dim, act_dim, num_agents, size):

        self.state_buf      = np.zeros(combined_shape(size, state_dim*num_agents), dtype=np.float32)
        self.act_buf        = np.zeros(combined_shape(size, act_dim*num_agents), dtype=np.float32)
        self.rew_buf        = np.zeros(size, dtype=np.float32)  #rewards
        self.next_state_buf = np.zeros(combined_shape(size, state_dim*num_agents), dtype = np.float32)
        #self.done_buf       = np.zeros(combined_shape(size, num_agents), dtype = np.float32) 
        self.ptr, self.max_size = 0, size

    def store(self, state, act, rew, next_state):
        """
        Append a single timestep to the buffer. This is called at each environment
        update to store the outcome observed outcome.
        """
        # buffer has to have room so you can store
        assert self.ptr < self.max_size

        self.state_buf[self.ptr,:] = state
        self.act_buf[self.ptr,:] = act
        self.rew_buf[self.ptr] = rew
        self.next_state_buf[self.ptr,:] = next_state
        
        self.ptr += 1

    def get(self):
        """
        Call after an epoch ends. Resets pointers and returns the buffer contents.
        """
        # Buffer has to be full before you can get something from it.
        assert self.ptr == self.max_size
        self.ptr =  0
        data = dict(state=self.state_buf, act=self.act_buf, rew=self.rew_buf,next_state =self.next_state_buf)

        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

