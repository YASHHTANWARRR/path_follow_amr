"""Train a SAC policy on the headless warehouse env, then report success rate and a learning curve."""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ament_index_python.packages import get_package_share_directory
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from warehouse_env import WarehouseEnv

LOOP_EPISODE_S = 150  # 1500 steps at 10 Hz
LOOP = [(-4.5, -3.0), (4.5, -3.0), (4.5, 7.0), (-4.5, 7.0)]  # closed rectangle, clearance 0.61 m


def evaluate(model, env, episodes, seed, loop):
    successes, laps, lap_times = 0, 0, []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc
            if "lap_steps" in info:
                lap_times.append(info["lap_steps"] * 0.1)
        successes += int(info.get("success", False))
        laps += info.get("laps", 0)
    if loop:
        best = min(lap_times) if lap_times else float("nan")
        mean = sum(lap_times) / len(lap_times) if lap_times else float("nan")
        print(f"loop: {laps / episodes:.2f} laps per {LOOP_EPISODE_S:.0f}s episode, best lap {best:.1f}s, mean lap {mean:.1f}s")
    return successes / episodes


def plot_curve(monitor_csv, out_png):
    rows = [r for r in csv.reader(open(monitor_csv)) if r and not r[0].startswith("#")][1:]
    rewards = np.array([float(r[0]) for r in rows])
    window = max(10, len(rewards) // 20)
    smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(7, 4))
    plt.plot(rewards, alpha=0.25, label="episode return")
    plt.plot(np.arange(len(smooth)) + window - 1, smooth, label=f"moving avg ({window})")
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.title("SAC on warehouse navigation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)


def main():
    default_map = os.path.join(get_package_share_directory("pathbot_navigation"), "maps", "warehouse.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=default_map)
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "models"))
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--loop", action="store_true", help="train the closed-loop lap task instead of goal reaching")
    ap.add_argument("--randomize", action="store_true", help="domain randomization: turn gain and scan noise")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    log_dir = os.path.join(args.out, "logs")
    os.makedirs(log_dir, exist_ok=True)
    waypoints = LOOP if args.loop else None
    env = Monitor(WarehouseEnv(args.map, waypoints=waypoints, randomize=args.randomize),
                  filename=os.path.join(log_dir, "monitor"))
    model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=100_000, batch_size=256,
                learning_starts=2_000, gamma=0.99, seed=0, device="cpu", verbose=1)
    model.learn(total_timesteps=args.steps)
    model_path = os.path.join(args.out, "sac_warehouse")
    model.save(model_path)

    plot_curve(os.path.join(log_dir, "monitor.monitor.csv"), os.path.join(log_dir, "learning_curve.png"))
    eval_env = WarehouseEnv(args.map, waypoints=waypoints, randomize=args.randomize)
    rate = evaluate(model, eval_env, args.eval_episodes, seed=10_000, loop=args.loop)  # held-out
    print(f"saved {model_path}.zip  held-out success rate: {rate:.1%} over {args.eval_episodes} episodes")


if __name__ == "__main__":
    main()
