import math, sys, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from stable_baselines3 import SAC
sys.path.insert(0, "/home/hornet/Desktop/pathfollowingbot/src/pathbot_rl/scripts")
from warehouse_env import build_obs, N_BEAMS, MAX_RANGE, MAX_V, MAX_W

class Probe(Node):
    def __init__(self):
        super().__init__("obs_probe"); self.scan = None; self.pose = None
        self.create_subscription(LaserScan, "/scan_fixed", lambda m: setattr(self, "scan", m), qos_profile_sensor_data)
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", lambda m: setattr(self, "pose", m), 10)

rclpy.init(); n = Probe()
while (n.scan is None or n.pose is None) and rclpy.ok():
    rclpy.spin_once(n, timeout_sec=0.5)
r = np.nan_to_num(np.asarray(n.scan.ranges, np.float32), nan=MAX_RANGE, posinf=MAX_RANGE)
print(f"scan: {len(n.scan.ranges)} beams, angle_min {n.scan.angle_min:.3f}, inc {n.scan.angle_increment:.5f}, "
      f"range min {np.nanmin(r):.2f} max {np.nanmax(r):.2f}, inf/nan count {int(np.sum(~np.isfinite(np.asarray(n.scan.ranges))))}")
pz = n.pose.pose.pose; q = pz.orientation
yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
print(f"amcl pose: x {pz.position.x:.3f} y {pz.position.y:.3f} yaw {yaw:.3f}")
grp = np.clip(r[:720], 0, MAX_RANGE).reshape(N_BEAMS, 20).min(axis=1)
print("policy beams (36):", np.round(grp, 2))
obs = build_obs(grp, 3.0, 0.0, 0.0, 0.0)  # same goal and zero-velocity state as the failed run
print("obs ranges mean/min/max:", round(float(obs[:36].mean()),3), round(float(obs[:36].min()),3), round(float(obs[:36].max()),3))
print("obs tail:", np.round(obs[36:], 3))
m = SAC.load("/home/hornet/Desktop/pathfollowingbot/src/pathbot_rl/models/sac_warehouse.zip", device="cpu")
print("action:", np.round(m.predict(obs, deterministic=True)[0], 3))
n.destroy_node(); rclpy.shutdown()
