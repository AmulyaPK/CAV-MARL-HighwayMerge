import torch
import gymnasium as gym
import highway_env
import numpy as np
from env_wrapper import MARLHighwayWrapper
from agents.mappo import MAPPO

def evaluate(num_episodes=30, render=True):
    # Create and wrap environment
    raw_env = gym.make("merge-v0", render_mode="human" if render else None)
    raw_env.unwrapped.configure({
        "action": {
            "type": "DiscreteMetaAction",
            "actions": ["LANE_LEFT", "LANE_RIGHT", "FASTER", "SLOWER"]
        }
    })
    env = MARLHighwayWrapper(raw_env)

    obs, info = env.reset()
    obs = np.array(obs)

    # Load trained MAPPO agent
    agent = MAPPO(state_dim=obs.shape[0], action_dim=env.action_space.n)
    agent.actor.load_state_dict(torch.load("checkpoints/mappo_shared_actor.pth"))
    agent.actor.eval()

    episode_rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if render:
                env.render()

        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}")

    env.close()
    print(f"\nDONE! Average Reward over {num_episodes} episodes: {np.mean(episode_rewards):.2f}")

if __name__ == "__main__":
    evaluate()