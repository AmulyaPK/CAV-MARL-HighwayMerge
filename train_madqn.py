import gymnasium as gym
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from agents.madqn import MADQN
from env_wrapper import MARLHighwayWrapper
import highway_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='merge-v0', help='HighwayEnv scenario')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    args = parser.parse_args()

    # Environment Setup
    raw_env = gym.make(args.env)
    raw_env.unwrapped.configure({
        "action": {
            "type": "DiscreteMetaAction",
            "actions": ["LANE_LEFT", "LANE_RIGHT", "FASTER", "SLOWER"]  # 4-action setup for compatibility
        }
    })
    env = MARLHighwayWrapper(raw_env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = MADQN(state_dim=state_dim, action_dim=action_dim, device=args.device)

    eps = 1.0
    eps_decay = 0.995
    eps_min = 0.05

    os.makedirs("results", exist_ok=True)
    episode_rewards, avg_rewards, collision_rate = [], [], []
    total_collisions = 0

    print(f"\nStarting MADQN training on {args.env} for {args.episodes} episodes\n")

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        crashed = False

        while not done:
            action = agent.act(obs, eps)
            next_obs, reward, done, info = env.step(action)
            agent.store(obs, action, reward, next_obs, float(done))
            agent.update()

            obs = next_obs
            ep_reward += reward
            crashed = crashed or info.get("crashed", False)

        eps = max(eps * eps_decay, eps_min)

        if crashed:
            total_collisions += 1

        episode_rewards.append(ep_reward)
        avg_rewards.append(np.mean(episode_rewards[-50:]))
        collision_rate.append(total_collisions / (ep + 1))

        if ep % 10 == 0:
            print(f"EP {ep:<4} | Reward: {ep_reward:>7.2f} | "
                  f"Avg(50): {avg_rewards[-1]:>7.2f} | "
                  f"CollRate: {collision_rate[-1]:.3f} | "
                  f"Eps: {eps:.3f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(agent.q.state_dict(), "checkpoints/madqn_shared_q.pth")
    print("\nDONE! Saved trained MADQN model to /checkpoints\n")

    np.savez("results/madqn_results.npz",
             rewards=episode_rewards,
             avg=avg_rewards,
             collisions=collision_rate)
    print("Saved training logs to results/madqn_results.npz")

    plt.figure(figsize=(8, 5))
    plt.plot(episode_rewards, color='lightgray', label="Episode Reward")
    plt.plot(avg_rewards, color='orange', linewidth=2, label="Smoothed (50ep Avg)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("MAPPO Training Progress — Ramp Merging")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/mappo_training_curve.png", dpi=300)
    plt.close()

    print("Saved training curve to results/madqn_training_curve.png\n")
    print("Training complete!")

if __name__ == '__main__':
    main()