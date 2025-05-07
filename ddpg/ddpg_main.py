# main.py
import argparse
import yaml

import robosuite as suite
from stable_baselines3.common.monitor import Monitor

from ddpg_interface import RobosuiteGymWrapper
from ddpg_build import make_ddpg

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def make_env_fn(cfg):
    def _init():
        env = suite.make(
            env_name=cfg["env"]["name"],
            robots=cfg["env"]["robots"],
            reward_shaping=cfg["env"]["reward_shaping"],
            reward_scale=cfg["env"]["reward_scale"],
            use_camera_obs=cfg["env"]["use_camera_obs"],
            has_renderer=cfg["env"]["has_renderer"],
            has_offscreen_renderer=cfg["env"]["has_offscreen_renderer"],
            control_freq=cfg["env"]["control_freq"],
        )
        env = RobosuiteGymWrapper(env)
        env = Monitor(env, cfg["logging"]["monitor_log"])
        return env
    return _init

def main(config_path):
    # Load YAML
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Build DDPG 
    env_fn = make_env_fn(cfg)
    model = make_ddpg(env_fn, cfg)

    # Train
    model.learn(total_timesteps=cfg["training"]["total_timesteps"])

    # Save
    model.save(cfg["model"]["save_path"])
    print(f"Model saved to: {cfg['model']['save_path']}.zip")
    
    # Plot training curves
    logdir = cfg["logging"]["monitor_log"]
    csvs = glob.glob(os.path.join(logdir, "*.csv"))
    if not csvs:
        print(f"No log CSV found in {logdir}, skipping plots.")
        return
    df = pd.read_csv(sorted(csvs)[-1], comment="#")

    # Episode Reward per Episode
    plt.figure()
    plt.plot(df["r"])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DDPG Episode Reward per Episode")
    plt.savefig("ddpg_episode_reward.png")

    # Smoothed Reward (50-episode MA)
    df["reward_ma"] = df["r"].rolling(window=50).mean()
    plt.figure()
    plt.plot(df["reward_ma"])
    plt.xlabel("Episode")
    plt.ylabel("Reward (MA50)")
    plt.title("DDPG Smoothed Reward (50-episode MA)")
    plt.savefig("ddpg_smoothed_reward.png")

    # Episode Length per Episode
    plt.figure()
    plt.plot(df["l"])
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.title("DDPG Episode Length per Episode")
    plt.savefig("ddpg_episode_length.png")

    # Episode Reward over Time
    plt.figure()
    plt.plot(df["t"], df["r"])
    plt.xlabel("Time (s)")
    plt.ylabel("Reward")
    plt.title("DDPG Episode Reward over Time")
    plt.savefig("ddpg_reward_over_time.png")

    print("Plots saved: ddpg_episode_reward.png, ddpg_smoothed_reward.png, ddpg_episode_length.png, ddpg_reward_over_time.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the YAML config file"
    )
    args = parser.parse_args()
    main(args.config)
