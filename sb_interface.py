import numpy as np
import robosuite as suite
from vision import sim_vision
import torch
from PIL import Image
import math
import random

# interace with stable baselines

class Sim:
    def __init__(self, params):
        reward_params = params["reward_function"]
        self.params = params
        self.use_cost = bool(reward_params["cost"])
        self.cost_scale = reward_params.get("cost_scale", 0.01)
        self.reward_for_raise = reward_params.get("raise", 1.0)

        self.has_renderer = True
        self.use_camera_obs = bool(params.get("human", False))

        # for taking pictures
        self.sim_vision = sim_vision.SimVision(use_sim_camera=self.use_camera_obs)

        self.env = None
        self._setup_env()

        self.mem_reward = 0
        self.state = None
        self.done = False
        self.initial_cube_z = 0

    def _setup_env(self):
        """
        Construct Robosuite Lift environment (self.env)
        """
        self.env = suite.make(
            env_name="Lift",
            robots="Kinova3",
            has_renderer=self.has_renderer,
            has_offscreen_renderer=True,
            use_camera_obs=self.use_camera_obs,
            horizon=1000,
            camera_names="sideview" if self.use_camera_obs else [],
            camera_heights=400,
            camera_widths=400,
        )
        self.reset()

    def reset(self, has_renderer=None, use_sim_camera=None):
        """
        Initialize environment
        """
        obs = self.env.reset()

        # set all 7 kinova joints to starting position
        desired_joint_positions = [math.pi, 0.0, 0.0, -math.pi / 2, 0.0, -math.pi / 2, 0]
        self.env.robots[0].set_robot_joint_positions(desired_joint_positions)

        # run 5 time steps to "settle" robot
        for _ in range(5):
            obs, _, _, _ = self.env.step([0] * 7)

        # take observation and convert to state
        self.initial_cube_z = obs['cube_pos'][2]
        self.state = self._process_obs(obs)


        self.done = False
        self.sim_vision.reset()

    def observe(self):
        """
        Return current observation
        """
        if not isinstance(self.state, np.ndarray) or self.state.shape != (10,):
            print("[WARN] Sim.observe() returned bad state. Using fallback zeros.")
            return np.zeros(10, dtype=np.float32)
        return self.state.astype(np.float32)

    def _process_obs(self, obs):
        """
        Converts robot's observation into state
        """
        eef_pos = obs['robot0_eef_pos'] # (x,y,z) position of gripper
        cube_pos = obs['cube_pos'] # (x,y,z) position of cube
        gripper_open = obs.get('gripper_qpos', np.zeros(1))[0:1] # is gripper open?
        return np.concatenate([eef_pos, cube_pos, cube_pos - eef_pos, gripper_open])  # shape (10,)

    def act(self, action, w_video=False):
        """
        Executes action in robosuite
        """
        obs, reward, terminated, info = self.env.step(action)

        self.state = self._process_obs(obs)
        reward = self._compute_reward(obs, action)
        self.mem_reward = reward
        self.done = reward == self.reward_for_raise

        if self.has_renderer or w_video:
            self.env.render()
        
        return obs, reward, terminated, info

    def _compute_reward(self, obs, action):
        cube_pos = obs['cube_pos']
        eef_pos = obs['robot0_eef_pos']
        gripper_to_cube = obs['gripper_to_cube_pos']
        z_diff = cube_pos[2] - self.initial_cube_z
        dist = np.linalg.norm(gripper_to_cube)

        # ----- reward components -----
        r_lift = float(z_diff > 0.01 and dist < 0.04) * self.reward_for_raise
        r_approach = 1.0 - np.tanh(5.0 * dist)  # closer = ~1, far = ~0
        r_align_z = np.exp(-20 * abs(gripper_to_cube[2]))  # closer in Z

        reward = r_lift + 0.5 * r_approach + 0.3 * r_align_z

        # ----- optional cost on large actions -----
        if self.use_cost and r_lift == 0:
            reward -= self.cost_scale * np.linalg.norm(action)

        return reward


    def reward(self):
        return torch.tensor(self.mem_reward, dtype=torch.float32)

    def take_photo(self):
        obs, _, _, _ = self.env.step([0] * 7)
        img = Image.fromarray(obs["sideview_image"], 'RGB')
        img.save(f'vision/data/Robosuite3/Images/sideview{random.random() * 1000:.2f}.png')
