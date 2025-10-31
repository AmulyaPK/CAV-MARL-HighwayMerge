# This is the training driver script for MAPPO algorithm

import gymnasium as gym
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from agents.mappo import MAPPO
from env_wrapper import MARLHighwayWrapper
import highway_env

def main():
    # We load the environment, wrap it using the translation from env_wrapper and configures and 4-action space
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='merge-v0', help='HighwayEnv scenario')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    args = parser.parse_args()

    raw_env = gym.make(args.env)
    raw_env.unwrapped.configure({
        "action": {
            "type": "DiscreteMetaAction",
            "actions": ["LANE_LEFT", "LANE_RIGHT", "FASTER", "SLOWER"]  # 4-action setup for stability
        }
    })
    env = MARLHighwayWrapper(raw_env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initializing the MAPPO agent - building the actor-critic neural network (actor chooses actions and critic estimated state value)
    agent = MAPPO(state_dim=state_dim, action_dim=action_dim, device=args.device)

    os.makedirs("results", exist_ok=True)
    episode_rewards, avg_rewards, collision_rate = [], [], []
    total_collisions = 0

    print(f"\nStarting MAPPO training on {args.env} for {args.episodes} episodes\n")

    # main training loop:
    # Interacts with the environment: selects actions, collects rewards, stores transitions
    # After each episode, updates the actor/critic network using PPO optimization
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        crashed = False

        while not done:
            action, logp, value = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.store(obs, action, logp, reward, float(done), value)
            obs = next_obs
            ep_reward += reward
            crashed = crashed or info.get("crashed", False)

        update_res = agent.update()
        if crashed:
            total_collisions += 1

        episode_rewards.append(ep_reward)
        avg_rewards.append(np.mean(episode_rewards[-50:]))
        collision_rate.append(total_collisions / (ep + 1))

        if ep % 10 == 0:
            print(f"EP {ep:<4} | Reward: {ep_reward:>7.2f} | "
                  f"Avg(50): {avg_rewards[-1]:>7.2f} | "
                  f"CollRate: {collision_rate[-1]:.3f}")

    # Saving results:
    # Saving model weights
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(agent.actor.state_dict(), "checkpoints/mappo_shared_actor.pth")
    torch.save(agent.critic.state_dict(), "checkpoints/mappo_shared_critic.pth")
    print("\nSaved trained MAPPO models to /checkpoints\n")
    # Saving metrics
    np.savez("results/mappo_results.npz",
             rewards=episode_rewards,
             avg=avg_rewards,
             collisions=collision_rate)
    print("Saved training logs to results/mappo_results.npz")

    plt.figure(figsize=(8, 5))
    plt.plot(episode_rewards, color='lightgray', label="Episode Reward")
    plt.plot(avg_rewards, color='blue', linewidth=2, label="Smoothed (50ep Avg)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("MAPPO Training Progress — Ramp Merging")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # Saving reward plot
    plt.savefig("results/mappo_training_curve.png", dpi=300)
    plt.close()

    print("Saved training curve to results/mappo_training_curve.png\n")
    print("DONE! Training complete!")

if __name__ == '__main__':
    main()
