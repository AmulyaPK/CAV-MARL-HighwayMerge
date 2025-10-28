import torch
import gymnasium as gym
import highway_env
import numpy as np
from env_wrapper import MARLHighwayWrapper
from agents.madqn import MADQN

def evaluate(num_episodes=5, render=True):
    raw_env = gym.make("merge-v0", render_mode="human" if render else None)
    env = MARLHighwayWrapper(raw_env)
    obs, info = env.reset()
    obs = np.array(obs)

    # Load trained MADQN agent
    agent = MADQN(state_dim=obs.shape[0], action_dim=env.action_space.n)
    agent.q.load_state_dict(torch.load("checkpoints/madqn_shared_q.pth"))
    agent.q.eval()

    episode_rewards = []
    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.q(obs_tensor)
                action = torch.argmax(q_values, dim=1).item()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if render:
                env.render()

        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}")

    env.close()
    print(f"\n✅ Average Reward over {num_episodes} episodes: {np.mean(episode_rewards):.2f}")

if __name__ == "__main__":
    evaluate()
