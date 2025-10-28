import sys, os
sys.path.append(os.path.dirname(__file__))
import gymnasium as gym
import highway_env
from env_wrapper import MARLHighwayWrapper

# 1. Create the base highway environment
env = gym.make("highway-v0")

# 2. Wrap it with our MARL wrapper
wrapper = MARLHighwayWrapper(env)

# 3. Reset environment
obs, info = wrapper.reset()

print("✅ Reset successful.")
print("Original observation shape:", env.observation_space)
print("Wrapped observation shape:", obs.shape)
print("First few obs values:", obs[:10])

# 4. Take one random step
action = wrapper.action_space.sample()
print("\nTaking random action:", action)

next_obs, reward, done, info = wrapper.step(action)

print("Next observation shape:", next_obs.shape)
print("Reward from compute_paper_reward():", reward)
print("Done flag:", done)
print("Info dict:", info)
