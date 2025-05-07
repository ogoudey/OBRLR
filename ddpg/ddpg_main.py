# main.py
import argparse
import yaml

import robosuite as suite
from stable_baselines3.common.monitor import Monitor

from ddpg_interface import RobosuiteGymWrapper
from ddpg_build import make_ddpg

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the YAML config file"
    )
    args = parser.parse_args()
    main(args.config)
