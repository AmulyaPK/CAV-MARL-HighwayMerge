# MARL_CAVs/agents/networks.py
import torch
import torch.nn as nn
import math

def mlp(input_dim, output_dim, hidden_sizes=(128,128,128), activation=nn.Tanh):
    layers = []
    last = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, h))
        layers.append(activation())
        last = h
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)

class SharedActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(128,128,128)):
        super().__init__()
        # output logits (discrete actions)
        self.net = mlp(state_dim, action_dim, hidden_sizes, activation=nn.Tanh)

    def forward(self, x):
        logits = self.net(x)
        return logits

class SharedCritic(nn.Module):
    def __init__(self, state_dim, hidden_sizes=(128,128,128)):
        super().__init__()
        self.net = mlp(state_dim, 1, hidden_sizes, activation=nn.Tanh)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (batch,)

# For DQN-style Q net (MADQN), use dueling architecture:
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(128,128,128)):
        super().__init__()
        self.feature = mlp(state_dim, hidden_sizes[-1], hidden_sizes[:-1], activation=nn.ReLU)
        self.adv = nn.Sequential(nn.Linear(hidden_sizes[-1], hidden_sizes[-1]),
                                 nn.ReLU(), nn.Linear(hidden_sizes[-1], action_dim))
        self.val = nn.Sequential(nn.Linear(hidden_sizes[-1], hidden_sizes[-1]),
                                 nn.ReLU(), nn.Linear(hidden_sizes[-1], 1))

    def forward(self, x):
        f = self.feature(x)
        a = self.adv(f)
        v = self.val(f)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q
