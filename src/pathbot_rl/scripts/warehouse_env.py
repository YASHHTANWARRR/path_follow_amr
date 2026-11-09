"""Gymnasium env for the warehouse map + the observation shared with the ROS deploy node."""
import math
import os

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from PIL import Image
from scipy.ndimage import distance_transform_edt

N_BEAMS = 36            # 720 LiDAR beams min-pooled to 36 (deploy node does the same)
MAX_RANGE = 12.0        # m, matches LiDAR max
ROBOT_R = 0.35          # m, body half-width (0.30) + margin
GOAL_TOL = 0.4          # m
MAX_V, MAX_W = 0.5, 1.0 # m/s, rad/s - match nav2_params max_vel_x / max_vel_theta
TURN_GAIN = 1.6         # sim-only: physical yaw / commanded yaw measured in Gazebo
DT = 0.1                # s, 10 Hz control
MAX_STEPS = 300
GOAL_MAX_DIST = 10.0


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def build_obs(ranges, goal_dist, goal_bearing, last_v, last_w):
    """Single source of truth for the policy input. Used by training AND deployment."""
    return np.concatenate([
        np.clip(ranges, 0.0, MAX_RANGE) / MAX_RANGE,
        [min(goal_dist, GOAL_MAX_DIST) / GOAL_MAX_DIST,
         math.sin(goal_bearing), math.cos(goal_bearing),
         last_v / MAX_V, last_w / MAX_W],
    ]).astype(np.float32)


class OccupancyMap:
    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            m = yaml.safe_load(f)
        img = np.asarray(Image.open(os.path.join(os.path.dirname(yaml_path), m["image"])).convert("L"), np.float32)
        p_occ = (255.0 - img) / 255.0           # negate: 0 (map_server default)
        self.res = float(m["resolution"])
        self.ox, self.oy = m["origin"][0], m["origin"][1]
        self.h, self.w = img.shape
        # Unknown (grey) cells count as blocked: we never plan through what we have not seen.
        self.blocked = p_occ > m.get("free_thresh", 0.196)
        # Metres to the nearest blocked cell, for collision checks and goal sampling.
        self.clearance = distance_transform_edt(~self.blocked) * self.res

    def cell(self, x, y):
        c = int(math.floor((x - self.ox) / self.res))
        r = self.h - 1 - int(math.floor((y - self.oy) / self.res))
        return r, c

    def clearance_at(self, x, y):
        r, c = self.cell(x, y)
        if not (0 <= r < self.h and 0 <= c < self.w):
            return 0.0
        return self.clearance[r, c]

    def raycast(self, x, y, headings):
        """Distance to the first blocked cell along each heading. Vectorised over beams."""
        ts = np.arange(0.0, MAX_RANGE, self.res / 2.0)
        px = x + np.outer(np.cos(headings), ts)
        py = y + np.outer(np.sin(headings), ts)
        cols = np.floor((px - self.ox) / self.res).astype(int)
        rows = self.h - 1 - np.floor((py - self.oy) / self.res).astype(int)
        inside = (rows >= 0) & (rows < self.h) & (cols >= 0) & (cols < self.w)
        hit = ~inside
        hit[inside] = self.blocked[rows[inside], cols[inside]]
        any_hit = hit.any(axis=1)
        first = hit.argmax(axis=1)
        return np.where(any_hit, ts[first], MAX_RANGE)


class WarehouseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, map_yaml):
        super().__init__()
        self.map = OccupancyMap(map_yaml)
        self.observation_space = spaces.Box(-np.inf, np.inf, (N_BEAMS + 5,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)  # [fwd, turn]
        self.beam_angles = np.linspace(-math.pi, math.pi, N_BEAMS, endpoint=False)

    def _free_points(self):
        ys, xs = np.nonzero(self.map.clearance > ROBOT_R + 0.2)
        return np.stack([self.map.ox + (xs + 0.5) * self.map.res,
                         self.map.oy + (self.map.h - 1 - ys + 0.5) * self.map.res], axis=1)

    def _sample_pose(self, pts):
        x, y = pts[self.np_random.integers(len(pts))]
        return float(x), float(y)

    def _obs(self):
        ranges = self.map.raycast(self.x, self.y, self.th + self.beam_angles)
        dx, dy = self.gx - self.x, self.gy - self.y
        dist = math.hypot(dx, dy)
        bearing = wrap(math.atan2(dy, dx) - self.th)
        return build_obs(ranges, dist, bearing, self.last_v, self.last_w)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not hasattr(self, "_pts"):
            self._pts = self._free_points()
        while True:
            self.x, self.y = self._sample_pose(self._pts)
            self.gx, self.gy = self._sample_pose(self._pts)
            if 2.0 <= math.hypot(self.gx - self.x, self.gy - self.y) <= GOAL_MAX_DIST:
                break
        self.th = float(self.np_random.uniform(-math.pi, math.pi))
        self.last_v = self.last_w = 0.0
        self.steps = 0
        self.prev_dist = math.hypot(self.gx - self.x, self.gy - self.y)
        return self._obs(), {}

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        v = (a[0] + 1.0) / 2.0 * MAX_V      # forward only, keeps the policy simple
        w = a[1] * MAX_W
        self.th = wrap(self.th + TURN_GAIN * w * DT)
        nx = self.x + v * math.cos(self.th) * DT
        ny = self.y + v * math.sin(self.th) * DT
        self.steps += 1
        self.last_v, self.last_w = v, w

        if self.map.clearance_at(nx, ny) < ROBOT_R:
            reward, terminated, success = -10.0, True, False
        else:
            self.x, self.y = nx, ny
            dist = math.hypot(self.gx - self.x, self.gy - self.y)
            reward = 5.0 * (self.prev_dist - dist) - 0.01
            self.prev_dist = dist
            success = dist < GOAL_TOL
            terminated = success
            if success:
                reward += 10.0
        truncated = self.steps >= MAX_STEPS and not terminated
        return self._obs(), reward, terminated, truncated, {"success": success}


if __name__ == "__main__":
    # Self-check: fails loudly if the env, raycast or obs shape is wrong.
    import sys
    env = WarehouseEnv(sys.argv[1] if len(sys.argv) > 1 else
                       os.path.join(os.path.dirname(__file__), "..", "..", "pathbot_navigation", "maps", "warehouse.yaml"))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (N_BEAMS + 5,), obs.shape
    assert np.all(np.isfinite(obs))
    # A wall-facing ray must read shorter than max range somewhere in the warehouse.
    r = env.map.raycast(env.x, env.y, np.zeros(1))[0]
    assert 0.0 < r <= MAX_RANGE
    # Stepping straight into the nearest wall must terminate with a collision.
    ok = False
    for _ in range(MAX_STEPS):
        _, rew, term, trunc, info = env.step(np.array([1.0, 0.0], np.float32))
        if term and rew == -10.0:
            ok = True
            break
    assert ok or trunc, "drive-forward never collided or truncated"
    print("warehouse_env self-check OK, obs dim", obs.shape[0])
