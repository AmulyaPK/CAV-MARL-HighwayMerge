"""
env_wrapper.py defines a custom wrapper around the Highway Env environment to 
make it compatible with MARL algorithsm and the reward definitions from the 
research paper.
It takes the base environment from HighwayEnv and redefines what each observation,
action, and reward looks like for our agents.

Basically, this file translates the raw simulator data into the format, reward
and structure required by the MARL algorithms in the paper - MAPPO and MADQN.
"""

import gymnasium as gym
import numpy as np

# constants from paper
STATE_DIM = 25
ACTION_DIM = 4
COMM_RANGE = 200.0    # meters
V_DES = 33.0          # max speed limit (m/s) in paper table (or v_max)
V_THRE = 0.0          # speed threshold in eq (17) - use 0 or set as desired
TDES = 1.5            # desired time gap (t_h) from paper table
MIN_HEADWAY = 0.1     # fallback if no measured headway
# reward weights from paper (Eq 16 and Table 1)
ALPHA_R = 1.0
ALPHA_U = 20.0
ALPHA_S = 4.0
ALPHA_C = -1000.0
ALPHA_G = 100.0
W_P = 0.5
W_N = 0.5

class MARLHighwayWrapper(gym.ObservationWrapper):
    def __init__(self, env, state_dim=STATE_DIM, action_dim=ACTION_DIM, comm_range=COMM_RANGE):
        super().__init__(env)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.comm_range = comm_range
        # we return a flat 25-d vector
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.action_space = env.action_space  # keep underlying action space, but your algo uses action_dim

    def reset(self, **kwargs):
        # gym or gymnasium returns either obs or (obs, info); handle both
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            obs, info = out
            return self.observation(obs), info
        else:
            return self.observation(out)

    def observation(self, obs):
        """
        Here we extract fratuers like position, speed, heading and lane index of the ego
        vehicle and nearby vehicles. Then flattens this data into a 1D vector so that the 
        neural network can consume it easily.

        Convert underlying env observation into 25-d vector ordered as:
        [EV (5 features), FV (4f), SV (4f), PV (4f), LV (4f)] flattened and padded.
        Underlying obs is expected as an array shape (N, F) or dict — we handle arrays.
        Each vehicle features used: [lane_id, x, y, vx, vy] (or fallback).
        If a role is missing, the corresponding slot is zeros.
        """
        # Accept either structured obs (array) or dict from custom env
        try:
            arr = np.array(obs, dtype=np.float32)
        except Exception:
            # if obs is dict or other, try to get 'vehicles' key or similar
            arr = None

        features = []

        # helper to append vehicle features safely
        def append_vehicle(v):
            # v may be array-like with columns: x, y, vx, vy, presence or lane id etc.
            # Normalize to 5 values: [lane, x, y, vx, vy]
            if v is None:
                features.extend([0.0]*5)
                return
            v = np.array(v, dtype=np.float32)
            # if v has 5+ columns, take first 5 or map appropriately
            if v.size >= 5:
                # If original order is (x, y, vx, vy, present), try to map:
                # we will construct lane id fallback to 1.0 if y > 0 else 2.0 (simple)
                lane = 1.0
                x = float(v[0])
                y = float(v[1])
                vx = float(v[2])
                vy = float(v[3])
                features.extend([lane, x, y, vx, vy])
            elif v.size == 4:
                # assume x,y,vx,vy
                lane = 1.0
                features.extend([lane, float(v[0]), float(v[1]), float(v[2]), float(v[3])])
            else:
                features.extend([0.0]*5)

        # If obs is 2D (vehicle rows), assume ego first
        if arr is not None and arr.ndim == 2 and arr.shape[0] >= 1:
            # ego
            append_vehicle(arr[0])
            # neighbors: we need to select roles FV, SV, PV, LV by semantic ordering.
            # The paper groups vehicles into EV, FV (front on ramp), SV (rear on ramp),
            # PV (front on mainline), LV (rear on mainline). We approximate by sorting
            # by lane (y) and longitudinal position (x) relative to ego.
            ego_x = float(arr[0][0])
            ego_y = float(arr[0][1])
            # compute dx and lane flag
            neighbors = []
            for i in range(1, arr.shape[0]):
                v = arr[i]
                dx = float(v[0]) - ego_x
                lane = float(v[1])
                neighbors.append((dx, lane, v))
            # split by lane: assume lane value distinguishes ramp vs mainline
            ramp = [t for t in neighbors if t[1] != ego_y]  # differing y -> other lane
            main = [t for t in neighbors if t[1] == ego_y]
            # For ramp lane: find front (smallest positive dx) -> FV, rear (largest negative dx) -> SV
            FV = next((v for dx,ln,v in sorted(ramp, key=lambda x: x[0]) if dx>0), None)
            SV = next((v for dx,ln,v in sorted(ramp, key=lambda x: x[0], reverse=True) if dx<0), None)
            # For main lane: PV front, LV rear
            PV = next((v for dx,ln,v in sorted(main, key=lambda x: x[0]) if dx>0), None)
            LV = next((v for dx,ln,v in sorted(main, key=lambda x: x[0], reverse=True) if dx<0), None)
            # Append in the paper's order: PV, LV, FV, SV (but earlier description used EV+4 neighbors)
            # We'll append EV then PV, LV, FV, SV to be consistent with some mapping
            append_vehicle(PV)
            append_vehicle(LV)
            append_vehicle(FV)
            append_vehicle(SV)
        else:
            # Non-array obs: fallback: zero-pad EV + 4 neighbors
            features.extend([0.0] * (5 * 5))

        # pad/truncate to state_dim
        arrf = np.array(features, dtype=np.float32)
        if arrf.size < self.state_dim:
            arrf = np.concatenate([arrf, np.zeros(self.state_dim - arrf.size, dtype=np.float32)])
        else:
            arrf = arrf[:self.state_dim]
        return arrf

    # compute components exactly per paper eq (17)-(21) and group eq (22)-(23)
    def compute_paper_reward(self, ego, neighbors, info, env_state_obs=None):
        """
        Implements the weighted reward described in the paper:
        - safety
        - efficiency
        - comfort
        using the a-weights.
        Inputs:
          - ego: object (vehicle) from unwrapped env or None
          - neighbors: list of neighbor vehicle objects (may contain None)
          - info: env info dict
          - env_state_obs: optional raw observation for positions if ego object missing
        Returns final scalar reward per paper: R_i with group-based combination.
        """
        import math
        # handle missing ego
        if ego is None:
            return 0.0

        # get kinematic values robustly: try attributes, else use env_state_obs fallback
        def get_attr(obj, name, default=0.0):
            if obj is None:
                return default
            return getattr(obj, name, getattr(obj, name, default))

        # speed magnitude v (use longitudinal speed if available)
        v = get_attr(ego, 'speed', None)
        if v is None:
            # try velocity components
            vx = get_attr(ego, 'vx', 0.0)
            vy = get_attr(ego, 'vy', 0.0)
            v = math.hypot(vx, vy)
        # eq (17): normalized speed reward
        v_max = V_DES  # use desired/max speed
        v_thre = V_THRE
        R_r = 0.0
        if v_max - v_thre != 0:
            R_r = (v - v_thre) / (v_max - v_thre)
            # clip 0..1
            R_r = max(0.0, min(1.0, R_r))

        # eq (18): urgency: R_u = exp(- (X - d_end)^2 / (20 * d_end))
        # X: longitudinal position of ego (x) relative to end-of-ramp; d_end: ramp length
        # We attempt to read ego.position or ego.x. Fallback values if missing.
        X = 0.0
        d_end = getattr(self.env.unwrapped, 'merge_end', None)
        if d_end is None:
            # default: merge_end at 400m (paper uses merge area 300-400m)
            d_end = 400.0
        # try ego position
        pos = getattr(ego, 'position', None)
        if pos is not None:
            X = pos[0]
        else:
            X = getattr(ego, 'x', 0.0)
        # compute R_u
        try:
            R_u = math.exp(- ((X - d_end) ** 2) / (20.0 * d_end))
        except Exception:
            R_u = 0.0

        # eq (19): safety reward based on time headway: R_s = min(log(d_h / (v * t_h)), 1)
        # compute min across observed neighbors front/back. We'll take smallest time-headway
        t_h = TDES  # desired time headway from paper
        min_th = 1e9
        # neighbors might be objects; compute longitudinal distance to ego
        ego_x = None
        pos = getattr(ego, 'position', None)
        if pos is not None:
            ego_x = pos[0]
        else:
            ego_x = getattr(ego, 'x', 0.0)
        for n in (neighbors or []):
            if n is None:
                continue
            npos = getattr(n, 'position', None)
            if npos is None:
                nx = getattr(n, 'x', None)
                if nx is None:
                    continue
                else:
                    n_x = nx
            else:
                n_x = npos[0]
            d_h = abs(n_x - ego_x)
            # avoid division by zero
            denom = max(v * t_h, 1e-3)
            try:
                th = math.log(max(d_h / denom, 1e-6))
            except Exception:
                th = -10.0
            # paper uses min(log(...), 1)
            val = min(th, 1.0)
            min_th = min(min_th, val)
        if min_th == 1e9:
            R_s = 0.0
        else:
            R_s = min_th

        # eq (20): collision indicator
        crashed = bool(info.get('crashed', False) or info.get('collision', False))
        R_c = 1.0 if crashed else 0.0

        # eq (21): goal/destination reached indicator
        reached = bool(info.get('arrived', False) or info.get('goal_reached', False))
        R_g = 1.0 if reached else 0.0

        # compose individual reward (Eq 16)
        R_i = (ALPHA_R * R_r) + (ALPHA_U * R_u) + (ALPHA_S * R_s) + (ALPHA_C * R_c) + (ALPHA_G * R_g)

        # neighbor (group) reward: average of neighbor individual rewards within communication range (Eq 22/23)
        neighbor_rewards = []
        for n in (neighbors or []):
            if n is None:
                continue
            npos = getattr(n, 'position', None)
            if npos is None:
                nx = getattr(n, 'x', None)
                if nx is None:
                    continue
                n_x = nx
            else:
                n_x = npos[0]
            if abs(n_x - ego_x) <= self.comm_range:
                # best-effort: compute neighbor individual reward approximately
                # We'll reuse the same formulas but with neighbor kinematics
                nv = getattr(n, 'speed', None)
                if nv is None:
                    nv = math.hypot(getattr(n, 'vx', 0.0), getattr(n, 'vy', 0.0))
                # R_r_n
                R_rn = 0.0
                if v_max - v_thre != 0:
                    R_rn = max(0.0, min(1.0, (nv - v_thre) / (v_max - v_thre)))
                # R_un: use neighbor X if available
                npos_attr = getattr(n, 'position', None)
                if npos_attr is not None:
                    nX = npos_attr[0]
                else:
                    nX = getattr(n, 'x', 0.0)
                try:
                    R_un = math.exp(- ((nX - d_end) ** 2) / (20.0 * d_end))
                except Exception:
                    R_un = 0.0
                # R_sn: approximate by same procedure but coarse: use distance to ego
                d_hn = abs(n_x - ego_x)
                denom_n = max(nv * t_h, 1e-3)
                try:
                    R_sn = min(math.log(max(d_hn / denom_n, 1e-6)), 1.0)
                except Exception:
                    R_sn = 0.0
                R_cn = 1.0 if bool(getattr(n, 'crashed', False) or info.get('crashed', False)) else 0.0
                R_gn = 1.0 if bool(getattr(n, 'arrived', False) or info.get('goal_reached', False)) else 0.0
                R_n = (ALPHA_R * R_rn) + (ALPHA_U * R_un) + (ALPHA_S * R_sn) + (ALPHA_C * R_cn) + (ALPHA_G * R_gn)
                neighbor_rewards.append(R_n)
        R_N = float(np.mean(neighbor_rewards)) if len(neighbor_rewards) > 0 else 0.0

        # final group-based reward (Eq 23)
        R_G = W_P * R_i + W_N * R_N
        return float(R_G)

    def step(self, action):
        # call the underlying env step
        out = self.env.step(action)
        # handle gym vs gymnasium style
        if isinstance(out, tuple) and len(out) == 5:
            next_obs, reward, terminated, truncated, info = out
            done = terminated or truncated
        else:
            # assume old style 4-tuple
            next_obs, reward, done, info = out

        # unwrap env to access vehicle objects
        base_env = getattr(self.env, "unwrapped", self.env)
        # find ego and neighbors robustly
        ego = getattr(base_env, 'vehicle', None)
        if ego is None:
            # controlled_vehicles used in recent versions
            cvs = getattr(base_env, 'controlled_vehicles', None)
            if cvs and len(cvs) > 0:
                ego = cvs[0]
        # neighbors via road API if available
        neighbors = []
        try:
            if ego is not None and hasattr(base_env, 'road'):
                neighbors = base_env.road.neighbour_vehicles(ego)
        except Exception:
            # fallback: try to decode from next_obs if possible
            neighbors = []

        # compute paper reward
        try:
            final_reward = self.compute_paper_reward(ego, neighbors, info, env_state_obs=next_obs)
        except Exception as e:
            # on errors, fallback to env reward
            print("Warning compute_paper_reward error:", e)
            final_reward = float(reward if reward is not None else 0.0)

        return self.observation(next_obs), final_reward, done, info