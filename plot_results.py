import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d

# Utility
def load_results(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return None
    data = np.load(file_path)
    return {
        "rewards": data["rewards"],
        "avg": data["avg"],
        "collisions": data["collisions"]
    }

# Load both agents' logs
mappo_file = "results/mappo_results.npz"
madqn_file = "results/madqn_results.npz"

mappo_data = load_results(mappo_file)
madqn_data = load_results(madqn_file)

if mappo_data is None or madqn_data is None:
    print("ERROR! Missing results files. Run training first.")
    exit()

# Smooth curves
def smooth(x, sigma=3):
    return gaussian_filter1d(x, sigma=sigma)

# Plot average reward curves
plt.figure(figsize=(10,6))
plt.plot(mappo_data["avg"], label="MADQN (Smoothed)", color="#4CAF50", linewidth=2)
plt.plot(madqn_data["avg"], label="MAPPO (Smoothed)", color="#FF9800", linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("MAPPO vs MADQN — Cooperative Ramp Merging Performance")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("results/reward_comparison.png", dpi=300)
plt.show()

# Plot collision rate
plt.figure(figsize=(10,5))
plt.plot(smooth(mappo_data["collisions"]), label="MADQN", color="#4CAF50", linewidth=2)
plt.plot(smooth(madqn_data["collisions"]), label="MAPPO", color="#FF9800", linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Collision Rate")
plt.title("Collision Rate Over Training")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("results/collision_rate.png", dpi=300)
plt.show()

print("DONE! Saved plots to results/ (reward_comparison.png, collision_rate.png)")