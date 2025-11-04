import gymnasium as gym
import numpy as np

class MARLHighwayWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.comm_range = 50  # communication radius for neighbor awareness

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info

    def step(self, action):
        res = self.env.step(action)
        if len(res) == 5:
            obs, reward, terminated, truncated, info = res
            done = terminated or truncated
        else:
            obs, reward, done, info = res

        ego = getattr(self.env, "vehicle", None)
        neighbors = getattr(self.env, "road", None)
        neighbor_vehicles = []
        if neighbors is not None:
            for v in self.env.road.vehicles:
                if v is not ego:
                    neighbor_vehicles.append(v)

        # Compute custom reward
        reward = self.compute_paper_reward(ego, neighbor_vehicles, info)
        return self._process_obs(obs), reward, done, info

    def _process_obs(self, obs):
        """Flatten observation array."""
        if isinstance(obs, dict):
            obs = obs.get("observation", obs)
        return np.array(obs).flatten()

    def compute_paper_reward(self, ego, neighbors, info):
        """Reward shaped using α-weights from the paper."""
        if ego is None:
            return 0.0

        # α weights from paper
        α_v = 0.4  # velocity alignment
        α_s = 0.3  # safety distance
        α_c = 0.3  # comfort

        reward = 0.0

        # 1️⃣ Speed reward — normalized by desired speed
        desired_v = self.env.config.get("desired_speed", 30)
        r_speed = 1 - abs(ego.speed - desired_v) / desired_v
        r_speed = np.clip(r_speed, 0, 1)

        # 2️⃣ Safety reward — penalize close neighbors
        r_safety = 1.0
        for n in neighbors:
            dx = abs(n.position[0] - ego.position[0])
            dy = abs(n.position[1] - ego.position[1])
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 10:
                r_safety -= (10 - dist) / 10
        r_safety = np.clip(r_safety, 0, 1)

        # 3️⃣ Comfort reward — penalize harsh acceleration or yaw rate
        a_long = abs(getattr(ego, "acceleration", 0))
        a_lat = abs(getattr(ego, "yaw_rate", 0))
        r_comfort = 1 - min((a_long + a_lat) / 10, 1)

        # Weighted sum
        reward = α_v * r_speed + α_s * r_safety + α_c * r_comfort

        # Collision penalty
        if info.get("crashed", False):
            reward -= 5.0

        # Optional small scaling factor for training stability
        reward *= 10.0

        return float(reward)
