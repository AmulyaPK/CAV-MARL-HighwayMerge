import torch
import torch.nn.functional as F
import torch.optim as optim
from agents.networks import DuelingQNetwork
from agents.replay_buffer import ReplayBuffer
import numpy as np
from copy import deepcopy

class MADQN:
    def __init__(self, state_dim, action_dim, device='cpu',
                 lr=1e-3, gamma=0.99, buffer_capacity=200000, batch_size=256,
                 tau=0.005, hidden_sizes=(128,128,128)):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q = DuelingQNetwork(state_dim, action_dim, hidden_sizes).to(device)
        self.target_q = deepcopy(self.q)
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.buffer = ReplayBuffer(state_dim, action_dim, capacity=buffer_capacity)
        self.batch_size = batch_size
        self.tau = tau
        self.step = 0

    def act(self, state, eps=0.1):
        # epsilon greedy with shared network
        if np.random.rand() < eps:
            return np.random.randint(0, self.action_dim)
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qvals = self.q(s)
        return int(qvals.argmax().item())

    def update(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return

        batch = self.buffer.sample(batch_size)
        
        # Convert numpy arrays → torch tensors
        states = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device)

        # Q-value updates
        q_vals = self.q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_vals = self.target_q(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_vals * (1 - dones)

        loss = self.criterion(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        for param, target_param in zip(self.q.parameters(), self.target_q.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return loss.item()


    def store(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)
