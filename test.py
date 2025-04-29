import os
os.environ["MUJOCO_GL"] = "glfw"  # Must be set before importing robosuite

import robosuite
from robosuite import make
import numpy as np
import time

env = make(
    env_name="Lift",
    robots="Kinova3",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    horizon=200
)

obs = env.reset()
action_low, action_high = env.robots[0].controller.control_limits
action_dim = env.robots[0].controller.control_dim

for _ in range(1000):
    action = np.random.uniform(action_low, action_high, size=action_dim)
    obs, reward, done, _ = env.step(action)
    if done:
        obs = env.reset()
    time.sleep(0.01)
