# MARL_CAVs/agents/mappo.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from agents.networks import SharedActor, SharedCritic
import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.__init__()

class MAPPO:
    def __init__(self, state_dim, action_dim, device='cpu',
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, eps_clip=0.2, 
                 k_epochs=5, batch_size=8192, hidden_sizes=(128,128,128)):
        self.device = device
        self.actor = SharedActor(state_dim, action_dim, hidden_sizes).to(device)
        self.critic = SharedCritic(state_dim, hidden_sizes).to(device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.buffer = RolloutBuffer()
        self.batch_size = batch_size
        self.action_dim = action_dim

    def select_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.actor(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        value = self.critic(state_t)
        return int(action.item()), float(logp.item()), float(value.item())

    def store(self, state, action, logp, reward, done, value):
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(logp)
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)
        self.buffer.values.append(value)

    def compute_returns_and_advantages(self, last_value=0):
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values + [last_value]
        gae = 0
        advs = []
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step+1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * 0.95 * (1 - dones[step]) * gae  # lambda=0.95
            advs.insert(0, gae)
            returns.insert(0, gae + values[step])
        advs = torch.tensor(advs, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        # normalize adv
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        return returns, advs

    def update(self):
        if len(self.buffer.states) == 0:
            return None
        returns, advs = self.compute_returns_and_advantages()
        states = torch.tensor(self.buffer.states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.int64).to(self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        for _ in range(self.k_epochs):
            for b_states, b_actions, b_old_logps, b_returns, b_advs in loader:
                logits = self.actor(b_states)
                dist = torch.distributions.Categorical(logits=logits)
                logps = dist.log_prob(b_actions)
                ratio = torch.exp(logps - b_old_logps)
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_advs
                actor_loss = -torch.min(surr1, surr2).mean()
                # critic update
                values = self.critic(b_states)
                critic_loss = F.mse_loss(values, b_returns)
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()
                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

        self.buffer.clear()
        return total_actor_loss, total_critic_loss
