# MARL_CAVs/train_madqn.py
import gymnasium as gym, argparse, os
import numpy as np
from agents.madqn import MADQN
from env_wrapper import MARLHighwayWrapper
import torch
import highway_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='merge-v0')
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    # create env (you will need to register or use existing highway_env config)
    raw_env = gym.make("merge-v0")
    env = MARLHighwayWrapper(raw_env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = MADQN(state_dim=state_dim, action_dim=action_dim, device=args.device)
    eps = 1.0
    eps_decay = 0.995
    eps_min = 0.05

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.act(obs, eps)
            next_obs, reward, done, info = env.step(action)
            agent.store(obs, action, reward, next_obs, float(done))
            loss = agent.update()
            obs = next_obs
            ep_reward += reward
        eps = max(eps*eps_decay, eps_min)
        if ep % 10 == 0:
            print(f"EP {ep} reward {ep_reward:.2f} buffer {len(agent.buffer)}")
    # save model
    torch.save(agent.q.state_dict(), 'madqn_shared_q.pth')

if __name__ == '__main__':
    main()
