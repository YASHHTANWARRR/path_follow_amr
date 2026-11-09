#!/usr/bin/env python3
"""Runs the trained SAC policy on the Gazebo robot.

Inputs : /scan_fixed (LaserScan), /amcl_pose (map frame), /goal_pose (RViz 2D Goal Pose)
Output : /cmd_vel (Twist), zero when there is no goal, no pose, or the goal is reached.
"""
import math
import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from stable_baselines3 import SAC

from warehouse_env import GOAL_TOL, MAX_RANGE, MAX_V, MAX_W, N_BEAMS, build_obs, wrap

GROUP = 720 // N_BEAMS  # 720 LiDAR beams -> 36 policy beams, min-pooled


class RLController(Node):
    def __init__(self):
        super().__init__("rl_controller")
        default_model = os.path.join(get_package_share_directory("pathbot_rl"), "models", "sac_warehouse.zip")
        self.declare_parameter("model", "")
        model_path = self.get_parameter("model").value or default_model
        self.policy = SAC.load(model_path, device="cpu")
        self.ranges = None
        self.pose = None          # (x, y, yaw)
        self.goal = None          # (x, y)
        self.last_v = self.last_w = 0.0  # m/s and rad/s, same units as training
        self.create_subscription(LaserScan, "/scan_fixed", self.on_scan, qos_profile_sensor_data)
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self.on_pose, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self.on_goal, 10)
        self.pub = self.create_publisher(Twist, "/cmd_vel_nav", 10)  # Nav2 chain input: velocity_smoother -> collision_monitor -> /cmd_vel
        self.create_timer(0.1, self.control)  # 10 Hz, same as training DT
        self.get_logger().info(f"loaded policy {model_path}; set a goal in RViz")

    def on_scan(self, msg):
        r = np.nan_to_num(np.asarray(msg.ranges, np.float32), nan=MAX_RANGE, posinf=MAX_RANGE)
        r = np.clip(r[: GROUP * N_BEAMS], 0.0, MAX_RANGE)
        self.ranges = r.reshape(N_BEAMS, GROUP).min(axis=1)

    def on_pose(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (p.x, p.y, yaw)

    def on_goal(self, msg):
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f"new goal ({self.goal[0]:.2f}, {self.goal[1]:.2f})")

    def control(self):
        cmd = Twist()
        if self.ranges is not None and self.pose is not None and self.goal is not None:
            x, y, th = self.pose
            dx, dy = self.goal[0] - x, self.goal[1] - y
            dist = math.hypot(dx, dy)
            if dist < GOAL_TOL:
                self.goal = None
                self.get_logger().info("goal reached")
            else:
                obs = build_obs(self.ranges, dist, wrap(math.atan2(dy, dx) - th), self.last_v, self.last_w)
                act, _ = self.policy.predict(obs, deterministic=True)
                act = np.clip(act, -1.0, 1.0)
                self.last_v = float((act[0] + 1.0) / 2.0 * MAX_V)  # same mapping as WarehouseEnv.step
                self.last_w = float(act[1] * MAX_W)
                cmd.linear.x, cmd.angular.z = self.last_v, self.last_w
        self.pub.publish(cmd)


def main():
    rclpy.init()
    node = RLController()
    try:
        rclpy.spin(node)
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
