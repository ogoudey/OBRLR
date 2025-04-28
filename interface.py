import numpy as np
import robosuite as suite
from robosuite.utils import transform_utils
#from vision import camera_utils as cu
from vision import sim_vision
import torch
from PIL import Image

import matplotlib.pyplot as plt
import time
import random
from collections import defaultdict
from itertools import product
import tqdm
import json

import matplotlib.pyplot as plt
import networkx as nx
import torch
import cv2
import os


class Sim:
    def __init__(self, params):
    
        reward_params = params["reward_function"]
        self.env, self.mem_reward, self.state = None, None, None
        # Initialize (reset)
        self.has_renderer = False
        
        # Vision
        self.sim_vision = sim_vision.SimVision(use_sim_camera=False) # Change to turn simulated YOLO
        
        #
        
        # FOR RAISE REWARD #
        self.initial_cube_z = 0
        #
        has_renderer = False
        use_sim_camera = False
        if "human" in params.keys():
            if bool(params["human"]):
                use_sim_camera = True
                has_renderer = True
        
        if "testing" in params.keys():
            if bool(params["testing"]):
                has_renderer = True
        self.reset(has_renderer=has_renderer, use_sim_camera=use_sim_camera)
        
        self.use_cost = bool(reward_params["cost"])
        if self.use_cost:
            self.cost_scale = reward_params["cost_scale"]
        self.reward_for_raise = reward_params["raise"]
        
        
        
          
    def reset(self, has_renderer=False, use_sim_camera=False):
        if use_sim_camera:
            self.env = suite.make(
                env_name="Lift",
                robots="Kinova3",
                has_renderer=has_renderer,
                horizon = 2000,
                has_offscreen_renderer=True,
                use_camera_obs=True,
                camera_heights=400,
                camera_widths=400,
                camera_names="sideview",
                camera_depths=True
            )
        else:
            if not self.env:
                self.env = suite.make(
                    env_name="Lift",
                    robots="Kinova3",
                    has_renderer=has_renderer,
                    horizon = 2000,
                    has_offscreen_renderer=True
                )
            else:
                self.env.has_renderer=has_renderer
                self.has_renderer = has_renderer
                self.env.reset()
        """
                has_offscreen_renderer=True,
                use_camera_obs=True,
                
                camera_heights=400,
                camera_widths=400,
                camera_names="sideview",
                camera_depths=True
        """
        
        # Initialize
        obs, _, _, _ = self.env.step([0,0,0,0,0,0,0])
        self.mem_reward = torch.tensor(self.raise_reward(obs), dtype=torch.float32)
        detection = self.sim_vision.detect(obs, self.env.sim, no_cap= not has_renderer)
        self.state = detection
        
        self.sim_vision.reset()
        
        # FOR CUBE Z        
        self.initial_cube_z = obs['cube_pos'][2]
        

        
    def observe(self):
        state = self.state
        normalized_state = state # i.e. not normalized
        return normalized_state

    def act(self, action, w_video=False):
        standardized_action = [action[0], action[1], action[2], 0, 0, 0, action[3]]
        #print("Standardized action:", standardized_action)
        obs, _, _, _ = self.env.step(standardized_action)
        
        detection = self.sim_vision.detect(obs, self.env.sim, no_cap = not w_video)
        
        self.state = detection
        #print("State:", self.state)
        #print("Actual:", obs['robot0_eef_pos'] + obs['cube_pos'] + obs['robot0_gripper_qpos'].mea)
        
        ### Update reward ###
        raise_reward = torch.tensor(self.raise_reward(obs), dtype=torch.float32)
        if not raise_reward == self.reward_for_raise:
            if self.use_cost:
                raise_reward -= self.torq_cost(action)
        self.mem_reward = raise_reward
        
        
    
        
        
    def reward(self):
        return self.mem_reward
    
    def torq_cost(self, action):
        action_norm = np.linalg.norm(action)
        cost = self.cost_scale * action_norm
        #print("Action cost:", cost)
        
        return cost
        
    def raise_reward(self, obs):
        eef_pos = obs['robot0_eef_pos']
        cube_pos = obs['cube_pos']
        z_diff = (cube_pos[2] - self.initial_cube_z)
        #print("Reward calculation:", cube_pos[2], "-", self.initial_cube_z, "=", z_diff, "> 0.01? (1 or 0.1) AND...")
        #print("\teef_pos - cube_pos == X:", eef_pos, "-", cube_pos, "=", np.linalg.norm((eef_pos - cube_pos)), "< 0.04?")
        delta = np.linalg.norm((eef_pos - cube_pos))
        if z_diff > .01 and delta < 0.04:
            return self.reward_for_raise
        else:
            return -0.1
        
    def take_photo(self, i):
        # For gathering data
        obs, _, _, _ = self.env.step([0,0,0,0,0,0,0])
        img = Image.fromarray(obs["sideview_image"], 'RGB')
        img_name = 'sideview'+str(i)+'.png'
        #print("Photo-taking turned off.")
        img.save('vision/data/Robosuite3/Images' + img_name)


if __name__ == "__main__":
    s = Sim()
    s.has_renderer = True
    s.reset()
    while True:
        action = np.array(json.loads(input("Action: ")))
        obs, _, _, _ = s.env.step(action)
        reward = torch.tensor(s.raise_reward(obs), dtype=torch.float32)
        print(torch.tensor([reward]))
        state = torch.tensor(np.concatenate((obs['robot0_eef_pos'], obs['cube_pos'])), dtype=torch.float32)
        print(state)
    
