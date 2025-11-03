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
        self.buffer = ReplayBuffer(buffer_capacity, state_dim, device)
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

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch['states'], batch['actions'], batch['rewards'], batch['next_states'], batch['dones']
        q_vals = self.q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double Q: use q for argmax then target_q for value
            next_actions = self.q(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_q(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * (1 - dones) * next_q
        loss = F.mse_loss(q_vals, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # soft update
        for p, tp in zip(self.q.parameters(), self.target_q.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
        self.step += 1
        return loss.item()

    def store(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)
