# SAC navigation policy: training and evaluation log

| Item | Value |
|------|-------|
| Algorithm | SAC (Stable-Baselines3 2.9.2, Torch 2.15 CPU, MlpPolicy) |
| Environment | `pathbot_rl/scripts/warehouse_env.py`, 2D kinematic sim on `warehouse.pgm` |
| Observation | 36 LiDAR ranges (min-pooled from 720) + goal distance, sin/cos bearing, last v, last w (41 dims) |
| Action | [forward, turn] in [-1, 1], v in [0, 0.5] m/s, w in [-1, 1] rad/s |
| Reward | 5 x progress - 0.01/step, +10 goal (within 0.4 m), -10 collision |
| Episode | max 300 steps, goal 2-10 m from start |
| Hyperparameters | lr 3e-4, buffer 100k, batch 256, learning_starts 2000, gamma 0.99 |
| Training steps | 150,000 (about 25 min, CPU) |
| Held-out success | **89.0%** over 100 episodes (goal positions not used in training, seed 10000+) |

Files: `train_sac_log.txt` (full training output), `monitor.monitor.csv` (per-episode return and length), `learning_curve.png`.

Known gaps: evaluation is in the 2D simulator only. Gazebo deployment results are recorded separately below.

## Gazebo deployment (attempt 1, FAILED to reach goal)

Setup: `bringup.launch.py` (AMCL mode) + `rl_controller.launch.py`; goal (3.0, 0.0) map frame.
Result: the robot did not reach the goal. Over 45 s it stayed within about 0.1 m of the start,
with mostly in-place turning. The controller received the goal and produced non-zero policy
actions in an offline check.

Fixed along the way: controller crash on empty model path; commands were overwritten by other
publishers on /cmd_vel, /cmd_vel_smoothed. Controller now publishes to /cmd_vel_nav (Nav2 chain input).

Likely causes, not yet tested: sim-to-sim gap (the trainer uses a 2D kinematic model with a
calibrated turn gain of 1.6 and perfect map; Gazebo has skid-steer slip, scan noise and a
different scan resolution), and the policy has not learned to handle clutter near the start.
Full log: `deploy_gazebo_log.txt`.

## Loop task (closed rectangle, lap time), domain randomized

Route: rectangle (-4.5,-3) -> (4.5,-3) -> (4.5,7) -> (-4.5,7), closest wall 0.61 m.
Training: 300k steps, randomized turn gain (1.2-2.0) and scan noise (sigma 0.03 m, 2% dropout),
reward = progress + 0.5*speed - 0.01/step + 10 per waypoint + 50 per lap, -10 on collision.
Episode: 1500 steps (150 s).

| Item | Value |
|------|-------|
| Final reward (ep_rew_mean) | 818 at 279k steps (rising from 26 at start) |
| Laps per 150 s episode (20 held-out episodes, simulator) | 2.00 |
| Best lap | 54.3 s |
| Mean lap | 64.3 s |

Lap-time trend across training is not measured yet (only final evaluation). Files: `train_loop_log.txt`,
`loop_learning_curve.png`, `loop_monitor.csv`.

### Gazebo loop test (attempt 1, not working yet)
The policy loads, the scan (10 Hz) and AMCL pose arrive, but `/cmd_vel_nav` carries zeros from
Nav2's `controller_server` and `behavior_server`, which publish on the same topic. The robot did not
move. Earlier goal-reaching success was timing-dependent. Next step: a clean command path for RL
(either remove Nav2's command publishers in RL mode or publish on a topic nothing else uses).
