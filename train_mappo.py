# MARL_CAVs/train_mappo.py
import gymnasium as gym, argparse, os
import numpy as np
from agents.mappo import MAPPO
from env_wrapper import MARLHighwayWrapper
import torch
import highway_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='highway-v0')
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    raw_env = gym.make("merge-v0")
    env = MARLHighwayWrapper(raw_env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = MAPPO(state_dim=state_dim, action_dim=action_dim, device=args.device)

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        # collect rollouts
        while not done:
            action, logp, value = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.store(obs, action, logp, reward, float(done), value)
            obs = next_obs
            ep_reward += reward
        update_res = agent.update()
        if ep % 10 == 0:
            print(f"EP {ep} reward {ep_reward:.2f} update {update_res}")
    torch.save(agent.actor.state_dict(), 'mappo_shared_actor.pth')
    torch.save(agent.critic.state_dict(), 'mappo_shared_critic.pth')

if __name__ == '__main__':
    main()
