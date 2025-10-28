import gymnasium as gym
import highway_env

env = gym.make("merge-v0", render_mode="human")
obs, info = env.reset()
print("✅ merge-v0 works! Observation shape:", env.observation_space.shape)
env.render()