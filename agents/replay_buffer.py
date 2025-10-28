# MARL_CAVs/agents/replay_buffer.py
import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, state_dim, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def push(self, s, a, r, s2, done):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s2
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            states = torch.tensor(self.states[idx]).to(self.device),
            actions = torch.tensor(self.actions[idx]).to(self.device),
            rewards = torch.tensor(self.rewards[idx]).to(self.device),
            next_states = torch.tensor(self.next_states[idx]).to(self.device),
            dones = torch.tensor(self.dones[idx]).to(self.device),
        )
        return batch

    def __len__(self):
        return self.size
